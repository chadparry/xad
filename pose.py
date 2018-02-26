import collections

Projection = collections.namedtuple('Projection', ['cameraIntrinsics', 'pose'])
CameraIntrinsics = collections.namedtuple('CameraIntrinsics', ['cameraMatrix', 'distCoeffs'])
Pose = collections.namedtuple('Pose', ['rvec', 'tvec'])
