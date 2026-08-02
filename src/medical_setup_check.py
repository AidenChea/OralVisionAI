import monai
import nibabel
import pydicom
import SimpleITK
import matplotlib

print("=" * 40)
print("Medical Imaging Environment Check")
print("=" * 40)

print("MONAI:", monai.__version__)
print("Nibabel:", nibabel.__version__)
print("pydicom:", pydicom.__version__)
print("SimpleITK:", SimpleITK.Version_VersionString())
print("Matplotlib:", matplotlib.__version__)

print("\n✅ Medical imaging setup passed!")
