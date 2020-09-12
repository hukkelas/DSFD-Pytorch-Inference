import pycuda.driver as cuda
import pycuda.autoinit
import os
import numpy as np
import cv2
import tensorrt as trt
from face_detection.retinaface.detect import RetinaNetDetectorONNX
from .utils import python_nms


class TensorRTRetinaFace:

    def __init__(
            self,
            imshape,  # (height, width)
            onnx_filepath: str = "detector_net.onnx"):
        if not os.path.isfile(onnx_filepath):
            detector = RetinaNetDetectorONNX(
                imshape)
            detector.export_onnx(onnx_filepath)
        self.TRT_LOGGER = trt.Logger(trt.tensorrt.Logger.Severity.INFO)
        self.engine_path = "model.engine"
        self.engine = self.build_engine(onnx_filepath)
        self.context = self.engine.create_execution_context()
        self.initialize_bindings()

    def initialize_bindings(self):
        self.input_bindings = []
        self.output_bindings = []
        for idx in range(self.engine.num_bindings):
            print(
                self.engine.get_binding_name(idx),
                self.engine.get_binding_dtype(idx),
                self.engine.get_binding_shape(idx))
            if self.engine.binding_is_input(idx):  # we expect only one input
                input_shape = self.engine.get_binding_shape(idx)
                input_size = trt.volume(input_shape) * self.engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
                self.input_bindings.append({
                    "input_shape": input_shape,
                    "input_size": input_size,
                    "device_input": cuda.mem_alloc(input_size),
                })
            else:  # and one output
                output_shape = self.engine.get_binding_shape(idx)
                host_output = cuda.pagelocked_empty(trt.volume(output_shape) * self.engine.max_batch_size, dtype=np.float32)
                device_output = cuda.mem_alloc(host_output.nbytes)
                self.output_bindings.append({
                    "output_shape": output_shape,
                    "host_output": host_output,
                    "device_output": device_output,
                    "name": self.engine.get_binding_name(idx)
                })

    def build_engine(self, onnx_filepath: str):
        if os.path.isfile(self.engine_path):
            with open(self.engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
                return engine

        builder = trt.Builder(self.TRT_LOGGER)
        network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_creation_flag)
        print(network)

        parser = trt.OnnxParser(network, self.TRT_LOGGER)
        # parse ONNX
        with open(onnx_filepath, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        print('Completed parsing of ONNX file')
        builder.max_batch_size = 1
        builder.debug_sync = True
        builder.max_workspace_size = 2**34

        if builder.platform_has_fast_fp16:
            builder.fp16_mode = True

        print('Building an engine...')
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")

        with open(self.engine_path, "wb") as f:
            f.write(engine.serialize())
        return engine

    def run_engine(self, img):
        stream = cuda.Stream()
        cuda.memcpy_htod_async(
            self.input_bindings[0]["device_input"], img, stream)
        bs = [int(x["device_input"]) for x in self.input_bindings] +\
             [int(x["device_output"]) for x in self.output_bindings]
        self.context.execute_async(
            bindings=bs,
            stream_handle=stream.handle)
        for out in self.output_bindings:
            cuda.memcpy_dtoh_async(
                out["host_output"], out["device_output"], stream)
            out["host_output"] = out["host_output"].reshape(out["output_shape"])
        assert len(self.output_bindings) == 1
        stream.synchronize()
        out = out["host_output"]
        assert out.shape[1] == 15
        boxes = out[:, :4]
        landms = out[:, 4:-1]
        scores = out[:, -1]
        return boxes, landms, scores

    def infer(self, img):
        confidence_t = np.array(0.3)
        nms_t = np.array(0.3)
        img = np.rollaxis(img, axis=-1)[None].astype(np.float32)
        img = np.ascontiguousarray(img).astype(np.float32)
        boxes, landms, scores = self.run_engine(img)
        keep_idx = scores >= confidence_t
        boxes, landms, scores = [_[keep_idx] for _ in [boxes, landms, scores]]
        keep_idx = python_nms(boxes, nms_t)
        boxes, landms, scores = [_[keep_idx] for _ in [boxes, landms, scores]]
        return boxes, landms, scores


if __name__ == "__main__":
    image = cv2.imread("images/0_Parade_marchingband_1_765.jpg")
    width = 640
    height = 480
    expected_imsize = (height, width)
    image = cv2.resize(image, (width, height))
    detector = TensorRTRetinaFace(
        (480, 640))
    print(detector.infer(image))
    boxes, landms, scores = detector.infer(image)
    for i in range(boxes.shape[0]):
        print(boxes[i])
        x0, y0, x1, y1 = boxes[i].astype(int)
        image = cv2.rectangle(image, (x0, y0), (x1, y1),(255, 0, 0), 1 )
    cv2.imwrite("test.png", image)