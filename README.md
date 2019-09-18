# The Lite Upsample Network Carafe unoffical
- This is an (unofficial) implementation of [CARAFE: Content-Aware ReAssembly of FEatures](https://arxiv.org/abs/1905.02188) with pytorch. 
- The network is a light upsample module which performs well and has less parameters. 

# Questions
- The implementation of Context aware Reassembly Module maybe not the best. Is there any better ways to generate stride features?

# Done
- The module has been tested on my own yolo and imporved about 1mAP on VOC 2007 test dataset. The code will be relesed later.

# TODO
- Check if bn and relu in Context aware Ressembly can imporve the module.
