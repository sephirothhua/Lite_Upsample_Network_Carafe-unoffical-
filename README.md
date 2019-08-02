# The Lite Upsample Network Carafe unoffical
- This is an (unofficial) implementation of [CARAFE: Content-Aware ReAssembly of FEatures](https://arxiv.org/abs/1905.02188) with pytorch. 
- The network is a light upsample module which performs well and has less parameters. 

# Questions
- The forward takes much time in Kernel Predeiction Module, maybe the kernel is too large?
- The implementation of Context aware Reassembly Module maybe not the best. Is there any better ways to generate stride features?

# TODO
- The module haven't be tested on any baseline yet. Testing....
