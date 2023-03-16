# Automatic Detection of iNPH using Deep Learning

## The pipeline:

* Segment lateral ventricles using Fastsurfer.
* Calculate AC/PC co-ordinates using HighRes3DNet.
* Find the plane parallel to AC-PC plane intersecting temporal horns of the lateral ventricles with maximum width.