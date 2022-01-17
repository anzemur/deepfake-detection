import cv2 as cv
import mediapipe as mp

def denormalize_face_position(img, face_loc):
  """
  Parse face position into the proporsions of the given image.

  Args:
    img: Image.
    face_loc: Normalized location of the face on the given image.

  Returns:
    Face location in proporsions to the given image.
  """

  img_h, img_w, _ = img.shape

  return {
    'x': int(face_loc['x'] * img_w),
    'y': int(face_loc['y'] * img_h),
    'height': int(face_loc['height'] * img_h),
    'width': int(face_loc['width'] * img_w)
  }


def crop_face(img, face_loc):
  """
  Crops face from the given image.

  Args:
    img: Image to crop face from.
    face_loc: Location of the face on the given image.

  Returns:
    Image with the cropped face.
  """

  return img[face_loc['y']:face_loc['y'] + face_loc['height'], face_loc['x']:face_loc['x'] + face_loc['width']]


def detect_face(img, denormalize=True):
  """
  Detects face in the given image.

  Args:
    img: Image to detect face in.
    denormalize: Tells if face location should be return in the ratio relative to the given image.

  Returns:
    Face location given by the top left coordinate and face width and height.
  """

  with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.8) as face_detection:
    results = face_detection.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    if not results.detections:
      return None

    for detection in results.detections:
      face_loc = detection.location_data.relative_bounding_box
      face_loc = {
        'x': face_loc.xmin,
        'y': face_loc.ymin,
        'height': face_loc.height,
        'width': face_loc.width
      }

      return denormalize_face_position(img, face_loc) if denormalize else face_loc


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
  """
  Expects a dlib face to generate a quadratic bounding box.
  
  Args:
    face: dlib face class
    width: frame width
    height: frame height
    scale: bounding box size multiplier to get a bigger face region
    minsize: set minimum bounding box size
  
  Returns:
    x, y, bounding_box_size in opencv form
  """
  x1 = face[0]
  y1 = face[1]
  x2 = face[2]
  y2 = face[3]

  size_bb = int(max(x2 - x1, y2 - y1) * scale)
  if minsize:
    if size_bb < minsize:
      size_bb = minsize
  center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

  # Check for out of bounds, x-y top left corner
  x1 = max(int(center_x - size_bb // 2), 0)
  y1 = max(int(center_y - size_bb // 2), 0)
  # Check for too big bb size for given x, y
  size_bb = min(width - x1, size_bb)
  size_bb = min(height - y1, size_bb)

  return {
    'x': x1,
    'y': y1,
    'height': size_bb,
    'width': size_bb
  }
