Spreadsheets defining MMMT model layer specifications such as their types, parameters, and connections to other layers. 

These documents contain the ground truth information independent of model input sizes. These spreadsheet are processed by M5 to derive the dataflow graph representation of the MMMT models using adjacency lists and to generate model configuration files specifying the graph node attributes (computation cost and layer type) along with edge weights (communication delays) for the given model input sizes and inter-FPGA communication bandwidth.
