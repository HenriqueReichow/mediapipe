#include <cstdlib>
#include <iostream>
#include <chrono>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#include "mediapipe/framework/formats/landmark.pb.h"

// Inclusão da shared memory
#include "shared/shared_data.h"

// Constantes do MediaPipe (streams e janela)
constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kLandmarksStream[] = "landmarks";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. If not provided, use webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path to save result (.mp4 only). If empty, show window.");

absl::Status RunMPPGraph(SharedData* shared) {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  ABSL_LOG(INFO) << "Loaded graph config";

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  cv::VideoCapture capture;
  const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (load_video) {
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                      graph.AddOutputStreamPoller(kOutputStream));

  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmark_poller,
                    graph.AddOutputStreamPoller(kLandmarksStream));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  bool grab_frames = true;
  while (grab_frames) {
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      if (!load_video) {
        ABSL_LOG(INFO) << "Ignoring empty frames from camera.";
        continue;
      }
      ABSL_LOG(INFO) << "Empty frame, end of video.";
      break;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, 1);
    }

    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    mediapipe::Packet packet;
    if (!poller.Next(&packet)) break;

    mediapipe::Packet landmark_packet;
    if (landmark_poller.QueueSize() > 0 && landmark_poller.Next(&landmark_packet)) {
      const auto& landmarks = landmark_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

      // Vamos enviar um comando para a memória compartilhada
      int comando_para_jogo = NONE;

      if (!landmarks.empty()) {
        // Exemplo: pegar o primeiro ponto da primeira mão para decidir o comando
        const auto& lm = landmarks[0].landmark(0);

        // Suponha lógica simples só para demo (x em [0,1]):
        if (lm.x() < 0.3) {
          comando_para_jogo = LEFT;
        } else if (lm.x() > 0.7) {
          comando_para_jogo = RIGHT;
        } else {
          comando_para_jogo = NONE;
        }
      }

      // Escrever na memória compartilhada
      shared->command = comando_para_jogo;
      shared->updated = 1;
    }

    auto& output_frame = packet.Get<mediapipe::ImageFrame>();
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

    if (save_video) {
      if (!writer.isOpened()) {
        ABSL_LOG(INFO) << "Abrindo arquivo de vídeo.";
        writer.open(absl::GetFlag(FLAGS_output_video_path),
                    mediapipe::fourcc('a', 'v', 'c', '1'),
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }
  }

  ABSL_LOG(INFO) << "Finalizando.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  // Inicializa a memória compartilhada
  SharedData* shared = init_shared_memory();
  shared->command = NONE;
  shared->updated = 0;

  absl::Status status = RunMPPGraph(shared);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Falha ao executar grafo: " << status.message();
    return EXIT_FAILURE;
  }
  ABSL_LOG(INFO) << "Sucesso!";
  return EXIT_SUCCESS;
}
