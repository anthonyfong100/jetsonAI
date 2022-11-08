import cv2
import argparse
import tritonclient.http as httpclient
from PIL import Image
from jetsonai.triton_client import TritonClientApi
from jetsonai.loaders import LocalFileLoader, WebCamLoader
from jetsonai.triton.model.enums import ClientType


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-a",
        "--async",
        dest="async_set",
        action="store_true",
        required=False,
        default=False,
        help="Use asynchronous inference API",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Use streaming inference API. "
        + "The flag is only available with gRPC protocol.",
    )
    parser.add_argument(
        "-m", "--model-name", type=str, required=True, help="Name of model"
    )
    parser.add_argument(
        "-x",
        "--model-version",
        type=str,
        required=False,
        default="",
        help="Version of model. Default is to use latest version.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=False,
        default=1,
        help="Batch size. Default is 1.",
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        required=False,
        default=1,
        help="Number of class results to report. Default is 1.",
    )
    parser.add_argument(
        "-s",
        "--scaling",
        type=str,
        choices=["NONE", "INCEPTION", "VGG"],
        required=False,
        default="NONE",
        help="Type of scaling to apply to image pixels. Default is NONE.",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-i",
        "--protocol",
        type=str,
        required=False,
        default="HTTP",
        help="Protocol (HTTP/gRPC) used to communicate with "
        + "the inference service. Default is HTTP.",
    )
    parser.add_argument(
        "image_filename",
        type=str,
        nargs="?",
        default=None,
        help="Input image / Input folder.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    FLAGS = parser.parse_args()
    concurrency = 20 if FLAGS.async_set else 1
    triton_client = httpclient.InferenceServerClient(
        url=FLAGS.url, verbose=FLAGS.verbose, concurrency=concurrency
    )
    triton_api = TritonClientApi(
        triton_client,
        ClientType.http,
        FLAGS.model_name,
        FLAGS.model_version,
        FLAGS.scaling,
        FLAGS.classes,
    )
    with WebCamLoader() as vid_stream:
        for _, frame in vid_stream.iter():
            resp = triton_api.infer(Image.fromarray(frame))
            cv2.imshow(vid_stream.window_name, frame)
            print(resp[0].class_name)
    # image_provider = LocalFileLoader(FLAGS.image_filename)
    # for image in image_provider.iter():
    #     resp = triton_api.infer(image)
    #     print(resp)
