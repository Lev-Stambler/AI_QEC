Currently "pretrained" on 10 rounds genetic with BP decoder.
I am unsure if this is the way to go though. BP OSD is defo slower though

Next steps: I think we also want to see what happens when we have *0* error on the stabilizers
and pass the PC matrix without the Checks in. This is somewhat stupid though as this is **equivalent** to a random code...
Maybe its worthwhile to see what happens if we apply this genetic technique to random classical codes? Perhaps to be honest thats a good idea.

Otherwise, maybe we have to have a decoder which *makes sense* for PC errors