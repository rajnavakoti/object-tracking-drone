import cv2
import time
from datetime import datetime
from threading import Thread

from deep_sort_realtime.deepsort_tracker import DeepSort
from djitellopy import Tello
from pyimagesearch.pid import PID

from YoloDetector import YoloDetector


# set run_id, track_person and keep_recording to TRUE
run_pid = True
track_person = True
keep_recording = True

# connect to drone and wait for 3 seconds
tello = Tello()
tello.connect()
time.sleep(3)

# switch on the video stream
tello.streamon()
frame_read = tello.get_frame_read()


def object_tracker():
    print("object tracking method has been called")
    pan_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    tilt_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    pan_pid.initialize()
    tilt_pid.initialize()
    max_speed_threshold = 40
    pan_update = 0
    tilt_update = 0

    detector = YoloDetector(model_name=None)
    obj_tracker = DeepSort(max_age=5,
                           n_init=2,
                           nms_max_overlap=1.0,
                           max_cosine_distance=0.3,
                           nn_budget=None,
                           override_track_class=None,
                           embedder="mobilenet",
                           half=True,
                           bgr=True,
                           embedder_gpu=True,
                           embedder_model_name=None,
                           embedder_wts=None,
                           polygon=False,
                           today=None)

    tello.send_rc_control(0, 0, 0, 0)

    while True:
        start = time.perf_counter()

        results = detector.score_frame(frame_read.frame)
        img, detections = detector.plot_boxes(results, frame_read.frame,
                                              height=frame_read.frame.shape[0],
                                              width=frame_read.frame.shape[1],
                                              confidence=0.5)

        print(img.shape[0])
        print(img.shape[1])

        tracks = obj_tracker.update_tracks(detections, frame=img)

        frame_center_x = img.shape[1] // 2
        frame_center_y = img.shape[0] // 2

        cv2.circle(img, (frame_center_x, frame_center_y), 20, (255, 0, 0), 2)

        # ---------------------------------------------------------------------
        for track in tracks:
            if not track.is_confirmed():
                tello.send_rc_control(0, 0, 0, 0)
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            bbox = ltrb

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

            obj_center = ((int(bbox[0]) + int(bbox[2])) / 2, (int(bbox[1]) + int(bbox[3])) / 2)
            x = int(obj_center[0])
            y = int(obj_center[1])
            cv2.circle(img, (x, y), 5, (255, 255, 255), -1)

            cv2.arrowedLine(img, (frame_center_x, frame_center_y), (x, y), color=(0, 255, 0), thickness=2)

            if run_pid:
                pan_error = frame_center_x - x
                pan_update = pan_pid.update(pan_error, sleep=0)

                tilt_error = frame_center_y - y
                tilt_update = tilt_pid.update(tilt_error, sleep=0)

                cv2.putText(img, f"X Error: {pan_error} PID: {pan_update:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(img, f"Y Error: {tilt_error} PID: {tilt_update:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255), 2, cv2.LINE_AA)

                if pan_update > max_speed_threshold:
                    pan_update = max_speed_threshold
                elif pan_update < -max_speed_threshold:
                    pan_update = -max_speed_threshold

                pan_update = pan_update * -1

                if tilt_update > max_speed_threshold:
                    tilt_update = max_speed_threshold
                elif tilt_update < -max_speed_threshold:
                    tilt_update = -max_speed_threshold

        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        cv2.putText(img, f'FPS: {int(fps)}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        print(int(pan_update), int(tilt_update))

        if track_person:
            print('trying to move the drone')
            tello.send_rc_control(int(pan_update / 4), 0, int(tilt_update / 2), 0)
            tello.send_rc_control(0, 0, 0, 0)
        # ----------------------------------------------------------------------

        cv2.imshow("Tracking screen", frame_read.frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()


def video_recorder():
    print("video recording method has been called")
    height, width, _ = frame_read.frame.shape
    video_file = f"video_{datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}.mp4"
    video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'h264'), 30, (width, height))

    while keep_recording:
        video.write(frame_read.frame)
        time.sleep(1 / 30)

    video.release()
    print("cv2 video released")


def terminate():
    global keep_recording

    keep_recording = False

    tello.land()
    print("Tello landed")

    recorder.join()
    print("Recording stopped")

    tello.streamoff()
    print("video stream is off")


recorder = Thread(target=video_recorder)

recorder.start()

print("upd address of Tello: ", tello.get_udp_video_address())
print("Tello battery: ", tello.get_battery())

tello.takeoff()
time.sleep(5)
print("Tello is flying", tello.is_flying)

if tello.is_flying:
    tello.move_up(50)
    time.sleep(3)
    print("Tello moved 100 up ")

    tello.rotate_counter_clockwise(360)
    time.sleep(6)
    print("Tello rotated clockwise")

    object_tracker()

    terminate()
else:
    terminate()

# try:
#     whatever()
# except Exception as e:
#     logging.error(traceback.format_exc())
#     # Logs the error appropriately.

# if __name__ == '__main__':
#     print_hi('PyCharm')
