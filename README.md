# Brian2Loihi implementation of the SpreizerNet

This is an implementation of the [Spreizer network](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007432) in the [Brian2Loihi Emulator](https://github.com/sagacitysite/brian2_loihi). It is based on [Leo Hiselius's Brian2 implementation of the Spreizer network](https://github.com/leohiselius/spreizer-net).

Relies on a local version of the Brian2loihi emulator, which is included.

Try out the jupyter notebook to see how it works.

Parameters can be changed in `loihi_spreizer_net/params`. The network is downsized to `1/4` of the original, i.e. `3600` excitatory neurons and `900` inhibitory neurons like in [Michaelis, Lehr, & Tetzlaff, 2020](https://www.frontiersin.org/articles/10.3389/fnbot.2020.589532/full). 

Noise input to the network generates sequential activity (see video in `movies`).
