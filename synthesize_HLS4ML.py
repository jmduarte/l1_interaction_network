import os
import hls4ml
import tensorflow as tf
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QActivation
from qkeras import QDense, QConv1D, QConv2D, quantized_bits
from qkeras.autoqkeras.utils import print_qmodel_summary

from pathlib import Path
import pprint

import numpy as np
from sklearn.metrics import accuracy_score


###############################################
# Add Nicolo's fix to cloning layer in HLS keeping input precicion in output
#  https://github.com/nicologhielmetti/enet-script/blob/master/optimizers/clone_type_matching.py
#

from hls4ml.model.optimizer import OptimizerPass


class CloneTypeMatching(OptimizerPass):
    def match(self, node):
        return node.__class__.__name__ == "Clone"

    def transform(self, model, node):
        inode = node.get_input_node()
        in_out_type = inode.get_output_variable().type.precision
        for out_out_var in node.variables.values():
            out_out_var.type.precision = in_out_type
        return False


hls4ml.model.optimizer.register_pass("clone_type_matching", CloneTypeMatching)


######################################################################################################################
# Add Vladimit fix for Transpose type matching in InteractionNetworks


class TransposeTypeMatching(OptimizerPass):
    def match(self, node):
        return node.__class__.__name__ == "Transpose"

    def transform(self, model, node):
        inode = node.get_input_node()
        in_out_type = inode.get_output_variable().type.precision
        for out_out_var in node.variables.values():
            out_out_var.type.precision = in_out_type
        return False


hls4ml.model.optimizer.register_pass("transpose_type_matching", TransposeTypeMatching)


######################################################################################################################


def print_dict(d, indent=0):
    align = 20
    for key, value in d.items():
        print("  " * indent + str(key), end="")
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(":" + " " * (20 - len(key) - 2 * indent) + str(value))


######################################################################################################################


def synthezise(synth=False):
    print("Synthesizing Model: {}".format(mname))

    # Get softmax layer name
    for i, layer in enumerate(model.layers):
        print(
            "Layer n=",
            i,
            "     class=",
            layer.__class__.__name__,
            "      name=",
            layer.name,
        )
        if layer.__class__.__name__ in ["Activation"]:
            cfg = layer.get_config()
            if cfg["activation"].find("softmax") != -1:
                softmax_name = layer.name
                print("{}: Tune hls4ml softmax implementation!".format(layer.name))

    # Make more QKeras compatible
    #  ap_ufixed<8, 0> just wrap around values that are out of bounds (not representable by <8,0>)
    #  while ap_ufixed<8, 0, AP_RND, AP_SAT>  saturate at the largest/smallest value representable by <8,0>.
    #  In C++ these distinct types that cannot be converted from one to another but we can easily imagine such conversion.
    hls4ml.model.optimizer.get_optimizer("output_rounding_saturation_mode").configure(
        layers=["Activation"]
    )
    hls4ml.model.optimizer.get_optimizer("output_rounding_saturation_mode").configure(
        rounding_mode="AP_RND_CONV"
    )
    hls4ml.model.optimizer.get_optimizer("output_rounding_saturation_mode").configure(
        saturation_mode="AP_SAT"
    )

    # Make HLS config
    config = hls4ml.utils.config_from_keras_model(
        model, granularity="name", default_reuse_factor=REUSE
    )  # , default_precision='ap_fixed<24,12>'

    #  config['Model']['Strategy'] = 'Resource'
    config["Model"]["Strategy"] = "Latency"
    config["LayerName"][softmax_name]["exp_table_t"] = "ap_fixed<18,8>"
    config["LayerName"][softmax_name]["inv_table_t"] = "ap_fixed<18,4>"

    # Handle large span of numerical values in input
    inputPrecision = "ap_fixed<20,10,AP_RND,AP_SAT>"
    for layer in model.layers:
        if layer.__class__.__name__ in ["BatchNormalization"]:
            config["LayerName"][layer.name]["Precision"]["scale"] = inputPrecision
            config["LayerName"][layer.name]["Precision"]["bias"] = inputPrecision
            config["LayerName"][layer.name]["Precision"]["result"] = inputPrecision
        if layer.__class__.__name__ in ["InputLayer"]:
            config["LayerName"][layer.name]["Precision"]["result"] = inputPrecision
        if layer.__class__.__name__ in ["QConv1D"]:  # For interaction network
            if "tmul" in layer.name and "linear" not in layer.name:
                config["LayerName"][layer.name]["Precision"]["weight"] = "ap_uint<1>"
                config["LayerName"][layer.name]["Precision"]["bias"] = "ap_uint<1>"

    # Add tracing to all hls model layers before adding non-traceable layers
    for layer in config["LayerName"].keys():
        config["LayerName"][layer]["Trace"] = True

    ###################################################################

    if "InteractionNetwork" in mname or "GraphConv" in mname:
        config["SkipOptimizers"] = ["reshape_stream"]
        if "InteractionNetwork" in mname:
            config["LayerName"][softmax_name]["Strategy"] = "Stable"
            config["LayerName"]["concatenate"] = {}
            config["LayerName"]["concatenate"]["Precision"] = inputPrecision
            config["LayerName"]["permute_1"] = {}
            config["LayerName"]["permute_1"]["Precision"] = inputPrecision
            config["LayerName"]["permute_2"] = {}
            config["LayerName"]["permute_2"]["Precision"] = inputPrecision
            config["LayerName"]["permute_3"] = {}
            config["LayerName"]["permute_3"]["Precision"] = inputPrecision

    ####################################################################

    # Bug! Cloned layer gets default precision rather than input precision TODO! Remove for new models fromAndre
    #  if 'QGraphConv' in mname:
    #    from hls4ml.model.optimizer.optimizer import optimizer_map
    #    optimizer_map.pop('clone_output')
    #    config['LayerName']['dense_1']['Precision'] = 'ap_fixed<14,5,AP_RND,AP_SAT>'

    ####################################################################
    # Special cases:

    changeStrategy = False
    if (
        changeStrategy
    ):  # Change strategy if layer is > 4,096. Doesn't work to set strategy per layer for io_stream models
        for layer in model.layers:
            config["LayerName"][layer.name]["Strategy"] = "Latency"
            w = layer.get_weights()[0]
            layersize = np.prod(w.shape)
            print("{}: {}".format(layer.name, layersize))  # 0 = weights, 1 = biases
            if layersize > 4096:  # assuming that shape[0] is batch, i.e., 'None'
                print(
                    "Layer {} is too large ({}), changing strategy Latency --> Resource".format(
                        layer.name, layersize
                    )
                )
                config["LayerName"][layer.name]["Strategy"] = "Resource"

    ####################################################################

    # Create the config
    print_dict(config)
    cfg = hls4ml.converters.create_config(FPGA_NAME)
    #  cfg = hls4ml.converters.create_config()

    # Set the config  IO mode to 'io_parallel' or 'io_stream'
    # if 'GraphConv' in mname or 'InteractionNetwork' in mname:
    if IO == "io_stream":
        print("USING IO STREAM !")
        cfg["IOType"] = "io_stream"
    else:
        print("USING IO PARALLEL !")
        cfg["IOType"] = "io_parallel"

    cfg["HLSConfig"] = config
    cfg["KerasModel"] = model
    cfg["OutputDir"] = "results/{}".format(mname)
    cfg["Part"] = FPGA_NAME  # new way of setting FPGA in master branch

    cfg["HLSConfig"]["LayerName"]["tmul_1"]["ReuseFactor"] = 1
    cfg["HLSConfig"]["LayerName"]["tmul_2"]["ReuseFactor"] = 1
    cfg["HLSConfig"]["LayerName"]["tmul_3"]["ReuseFactor"] = 1
    # Convert the Keras model to C++ and write it
    hls_model = hls4ml.converters.keras_to_hls(cfg)
    #  hls_model.write()

    # Compile the C++ model for bit-accurate CPU emulation and firmware/bitfile creation
    hls_model.compile()

    # Plot the NN model
    hls4ml.utils.plot_model(
        hls_model,
        show_shapes=True,
        show_precision=True,
        to_file="results/{}/hls_model_plot.png".format(mname),
    )
    tf.keras.utils.plot_model(
        model, to_file="results/{}/keras_model_plot.png".format(mname)
    )

    nfeat = 3
    X_test = np.load("./data/x_test_{}const.npy".format(NCONST))
    Y_test = np.load("./data/y_test_{}const.npy".format(NCONST), allow_pickle=True)

    if "MLP" in mname:
        X_test = np.reshape(X_test, (-1, NCONST * nfeat))
    else:
        X_test = np.ascontiguousarray(X_test)

    # Resize the input to 3000 entries
    X_test = X_test[:3000]
    Y_test = Y_test[:3000]

    print("##############################################")
    print("Model name : ", mname)
    print("Input shape X :", X_test.shape)
    print("Target shape X :", Y_test.shape)
    print("##############################################")

    # Get Keras and and HLS ( CPU bit-accurate emulation ) predictions
    y_hls = hls_model.predict(X_test)
    y_keras = model.predict(X_test)

    # Compare models accuracy. That was easy! Now let's see how the HLSHLS emulation  performance compares to Keras:
    print(
        "Keras  Accuracy: {}".format(
            accuracy_score(np.argmax(Y_test, axis=1), np.argmax(y_keras, axis=1))
        )
    )
    print(
        "hls4ml Accuracy: {}".format(
            accuracy_score(np.argmax(Y_test, axis=1), np.argmax(y_hls, axis=1))
        )
    )

    # Numerically profile of the model. This plots the distribution of the weights (and biases) as a box and whisker plot.
    #   The grey boxes show the values which can be represented with the data types used in the hls_model.
    #   Generally, you need the box to overlap completely with the whisker 'to the right' (large values) otherwise
    #   you'll get saturation & wrap-around issues. It can be okay for the box not to overlap completely 'to the left' (small values),
    #   but finding how small you can go is a matter of trial-and-error.
    """
  if not 'InteractionNetwork' in mname: # TODO! Add profiling for multiple inputs
    wp, wph, ap, aph = hls4ml.model.profiling.numerical(model,hls_model,X_test)
    wp.savefig("results/{}/wp.png".format(mname))
    wph.savefig("results/{}/wph.png".format(mname))
    ap.savefig("results/{}/ap.png".format(mname))
    aph.savefig("results/{}/aph.png".format(mname))
    fig = hls4ml.model.profiling.compare(model,hls_model,X_test)
    fig.savefig("results/{}/compare.png".format(mname))
   """

    # Synthesize the HLS model
    if synth:
        hls_model.build(csim=False, synth=True, vsynth=True)

        # Create  Vivado reports
        hls4ml.report.read_vivado_report("results/{}".format(mname))


#####################################################################################################################


# Function to print Vivado reports
def getReports(indir):
    data_ = {}

    report_vsynth = Path("results/{}/vivado_synth.rpt".format(indir))
    report_csynth = Path(
        "results/{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt".format(
            indir
        )
    )

    if report_vsynth.is_file() and report_csynth.is_file():
        print("Found valid vsynth and synth in {}! Fetching numbers".format(indir))

        # Get the resources from the logic synthesis report
        with report_vsynth.open() as report:
            lines = np.array(report.readlines())
            data_["lut"] = int(
                lines[np.array(["CLB LUTs*" in line for line in lines])][0].split("|")[
                    2
                ]
            )
            data_["ff"] = int(
                lines[np.array(["CLB Registers" in line for line in lines])][0].split(
                    "|"
                )[2]
            )
            data_["bram"] = float(
                lines[np.array(["Block RAM Tile" in line for line in lines])][0].split(
                    "|"
                )[2]
            )
            data_["dsp"] = int(
                lines[np.array(["DSPs" in line for line in lines])][0].split("|")[2]
            )
            data_["lut_rel"] = float(
                lines[np.array(["CLB LUTs*" in line for line in lines])][0].split("|")[
                    5
                ]
            )
            data_["ff_rel"] = float(
                lines[np.array(["CLB Registers" in line for line in lines])][0].split(
                    "|"
                )[5]
            )
            data_["bram_rel"] = float(
                lines[np.array(["Block RAM Tile" in line for line in lines])][0].split(
                    "|"
                )[5]
            )
            data_["dsp_rel"] = float(
                lines[np.array(["DSPs" in line for line in lines])][0].split("|")[5]
            )

        with report_csynth.open() as report:
            lines = np.array(report.readlines())
            lat_line = lines[
                np.argwhere(
                    np.array(["Latency (cycles)" in line for line in lines])
                ).flatten()[0]
                + 3
            ]
            data_["latency_clks"] = int(lat_line.split("|")[2])
            data_["latency_ns"] = float(lat_line.split("|")[2]) * 5.0
            data_["latency_ii"] = int(lat_line.split("|")[6])

    return data_


#########################################################################################################################


if __name__ == "__main__":

    models = [
        # "model_QMLP_nconst_8_nbits_4",
        # "model_QMLP_nconst_8_nbits_6",
        # "model_QMLP_nconst_8_nbits_8",
        # "model_QMLP_nconst_16_nbits_8",
        # "model_QMLP_nconst_32_nbits_8",
        # "model_QGraphConv_nconst_8_nbits_4",
        # "model_QGraphConv_nconst_8_nbits_6",
        # "model_QGraphConv_nconst_8_nbits_8",
        # "model_QGraphConv_nconst_16_nbits_4",
        # "model_QGraphConv_nconst_16_nbits_6",
        # "model_QGraphConv_nconst_16_nbits_8",
        # "model_QGraphConv_nconst_32_nbits_4",
        # "model_QGraphConv_nconst_32_nbits_6",
        # "model_QGraphConv_nconst_32_nbits_8",
        # "model_QInteractionNetwork_nconst_8_nbits_4",
        # "model_QInteractionNetwork_nconst_8_nbits_6",
        # "model_QInteractionNetwork_nconst_8_nbits_8",
        # "model_QInteractionNetwork_nconst_16_nbits_8",
        # "model_QInteractionNetwork_nconst_32_nbits_8",
        # "model_QInteractionNetwork_Conv1D_nconst_8_nbits_4",
        # "model_QInteractionNetwork_Conv1D_nconst_8_nbits_6",
        # "model_QInteractionNetwork_Conv1D_nconst_8_nbits_8",
        "model_QInteractionNetwork_Conv1D_nconst_16_nbits_4",
        # "model_QInteractionNetwork_Conv1D_nconst_16_nbits_8",
        # "model_QInteractionNetwork_Conv1D_nconst_32_nbits_4",
        # "model_QInteractionNetwork_Conv1D_nconst_32_nbits_8"
    ]

    # Set NCONSTIT in data and IO mode
    NCONST = 16

    # Set HLS4ML processing mode e reuse factor
    #  For Javier's CONV1D io_parallel hack (NCONST/REUSE) must be an integer !!!
    #  IO = 'io_stream'
    IO = "io_parallel"
    REUSE = 16

    # Set XILINX device
    # FPGA_NAME = 'xcvu9p-flgb2104-2l-e'
    FPGA_NAME = "xcvu9p-flgb2104-2L-e"

    # Synthesize and Print
    synth = True  # Synthesize the models
    prt = True  # Print Vivado reports ( latency and resource consumption )

    # Print Conda ENV
    print("Using CONDA ENVIRONMENT ------->", os.environ["CONDA_PREFIX"])

    # Loop over models to process
    for mname in models:
        model = tf.keras.models.load_model(
            "./models/{}.h5".format(mname),
            custom_objects={
                "QDense": QDense,
                "QActivation": QActivation,
                "QConv1D": QConv1D,
                "quantized_bits": quantized_bits,
            },
        )

        model.summary()
        print_qmodel_summary(model)

        # Synthesize model
        synthezise(synth=synth)

        # Print latency and resource consumption
        if prt:
            print("Reading hls project {}".format(mname))
            data = getReports("{}/".format(mname))
            print("\n Resource usage and latency: {}".format(mname))
            pprint.pprint(data)
