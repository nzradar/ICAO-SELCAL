# ICAO-SELCAL
A python script to decode ICAO SELCAL codes
This script takes input from a Windows(?) sounddevice (0 by default) and decodes the ICAO SELCAL tones into human readable format, namely:

20/02/26 19:36:31 AMBC

It will also log this information to a file called selcal_log.txt located in the same folder as the script.

There is also an option to lookup codes from a file called SELCAL_DICTIONARY.TXT, an example is provided.

Frequency information about the SELCAL tones is kept in a file called frequencies.json and this is provided for SELCAL 16 and SELCAL 32 tones.

No warranty is provided so have a crack at getting it to work better than I have. To be fair it's not bad but there are a lot of config tweaks at the top of the script to play with.
