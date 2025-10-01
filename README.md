A program to give a GUI to help guide through the filtering of DNA Jacks used in DNA Origami for the Nanoassembly Lab at UT Austin (Marras Research Group).

Primiarily used for James Houston's PhD Research.

GUI also incorporates OpenCV to detect jacks as well as estimate if it is properly folded, as well as the angle. Feature is currently unfinished.

Notes and random thoughts from 09/30/2025: for anyone picking this up, it's a bit disorganized. Ideally, I'd probably have used utility classes for the UI so it's not so complex in the main file, but honestly didn't expect to add so many things to it originally! 
I estimate most of the parameters using some math estimation and mostly crude tools. Before I stopped, I was thinking the solution is probably much simpler: stronger filtering and parameter adjustments with OpenCV could likely get pretty solidly closed contours on
most shapes, which would significantly reduce the other efforts needed to characterize the data. A lot of my code is working around the limitations of how clean the contours are (ex. merging nearby and overlapping contours). I also thought about using a basic data science model to help cateogrize the jacks, like a supervised learning approach e.g. kNN nearest neighbors (though kNN specifically is not likely a great approach in this case).

