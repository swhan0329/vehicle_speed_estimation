#!/usr/bin/env python

"""Vehicle speed estimation using the Lucas-Kanade tracker.

Usage:
    python main.py [video_source] [--output OUTPUT] [--show]
"""

from __future__ import print_function

import argparse
import sys

import cv2 as cv
import numpy as np

import video
from common import draw_str


class App:
    """Run Lucas-Kanade tracking and estimate lane speeds."""

    def __init__(self, video_src, output_path="output.mp4", show=False):
        self.track_len = 2
        self.detect_interval = 2
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.alpha = 0.5
        self.frame_idx = 0
        self.output_path = output_path
        self.show = show
        self.prev_gray = None

    def _create_writer(self, frame_size, fps):
        if not self.output_path:
            return None
        fourcc = cv.VideoWriter_fourcc(*("mp4v" if self.output_path.lower().endswith(".mp4") else "XVID"))
        writer = cv.VideoWriter(self.output_path, fourcc, fps, frame_size)
        if not writer.isOpened():
            print("Warning: unable to open output file: %s" % self.output_path)
            return None
        return writer

    def run(self):
        """Execute the tracking loop and estimate vehicle speeds."""
        if self.cam is None or not self.cam.isOpened():
            raise RuntimeError("Unable to open video source.")

        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        feature_params = dict(
            maxCorners=500,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7,
        )

        px2m = [0.0895, 0.088, 0.0774, 0.0767, 0.0736]
        ms2kmh = 3.6

        ret, frame = self.cam.read()
        if not ret or frame is None:
            raise RuntimeError("Unable to read first frame from source.")

        frame_h, frame_w = frame.shape[:2]
        fps = float(self.cam.get(cv.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 1.0:
            fps = 30.0

        writer = self._create_writer((frame_w, frame_h), fps)

        cal_mask = np.zeros((frame_h, frame_w), np.uint8)
        view_mask = np.zeros((frame_h, frame_w), np.uint8)
        view_polygon = np.array([[440, 1920], [420, 220], [680, 250], [1080, 480], [1080, 1920]], np.int32)
        cal_polygon = np.array([[440, 600], [420, 400], [1080, 400], [1080, 600]], np.int32)
        polygon1 = np.array([[550, 490], [425, 493], [420, 510], [570, 505]], np.int32)
        polygon2 = np.array([[565, 505], [555, 490], [680, 480], [720, 500]], np.int32)
        polygon3 = np.array([[680, 490], [690, 480], [800, 470], [800, 495]], np.int32)
        polygon4 = np.array([[840, 490], [820, 470], [950, 470], [960, 480]], np.int32)
        polygon5 = np.array([[1080, 480], [970, 480], [960, 470], [1080, 465]], np.int32)
        polygons = [polygon1, polygon2, polygon3, polygon4, polygon5]

        cv.fillConvexPoly(cal_mask, cal_polygon, 1)
        cv.fillConvexPoly(view_mask, view_polygon, 1)

        prv = [0.0] * 5
        prn = [0] * 5

        try:
            while True:
                vis = frame.copy()
                cmask = frame.copy()

                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame_gray = cv.bitwise_and(frame_gray, frame_gray, mask=cal_mask)
                vis = cv.bitwise_and(vis, vis, mask=view_mask)

                cv.line(vis, (400, 510), (1080, 475), (0, 0, 255), 5)
                cv.line(vis, (400, 495), (1080, 460), (0, 0, 255), 5)

                cv.fillPoly(cmask, [polygon1], (120, 0, 120), cv.LINE_AA)
                cv.fillPoly(cmask, [polygon2], (120, 120, 0), cv.LINE_AA)
                cv.fillPoly(cmask, [polygon3], (0, 120, 120), cv.LINE_AA)
                cv.fillPoly(cmask, [polygon4], (80, 0, 255), cv.LINE_AA)
                cv.fillPoly(cmask, [polygon5], (255, 0, 80), cv.LINE_AA)

                for lane_idx in range(5):
                    draw_str(vis, (30, 40 + lane_idx * 40), "%d-lane speed: %d km/h" % (lane_idx + 1, int(prv[lane_idx])))
                    draw_str(vis, (900, 40 + lane_idx * 40), "ptn%d: %d" % (lane_idx + 1, prn[lane_idx]))

                ptn = [0] * 5
                v = [0.0] * 5
                lane_total_dist = [0.0] * 5

                if self.tracks and self.prev_gray is not None:
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, _st, _err = cv.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, p0, None, **lk_params)
                    if p1 is not None:
                        p0r, _st, _err = cv.calcOpticalFlowPyrLK(frame_gray, self.prev_gray, p1, None, **lk_params)
                    else:
                        p0r = None

                    if p1 is not None and p0r is not None:
                        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                        good = d < 1
                        new_tracks = []
                        for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                            if not good_flag:
                                continue
                            tr.append((x, y))
                            if len(tr) > self.track_len:
                                del tr[0]
                            new_tracks.append(tr)
                            cv.circle(vis, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
                        self.tracks = new_tracks
                    else:
                        self.tracks = []

                    for tr in self.tracks:
                        if len(tr) < 2:
                            continue
                        dist = float(np.linalg.norm(np.subtract(tr[-1], tr[-2])))
                        for lane_idx, polygon in enumerate(polygons):
                            if cv.pointPolygonTest(polygon, tr[-1], False) > 0:
                                ptn[lane_idx] += 1
                                lane_total_dist[lane_idx] += dist

                    for lane_idx in range(5):
                        if ptn[lane_idx] > 0:
                            v[lane_idx] = (lane_total_dist[lane_idx] / ptn[lane_idx]) * px2m[lane_idx] * fps * ms2kmh

                    cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 0, 255))

                prn = ptn

                if self.frame_idx % self.detect_interval == 0:
                    for lane_idx in range(5):
                        if ptn[lane_idx] > 5:
                            prv[lane_idx] = v[lane_idx]
                            draw_str(
                                vis,
                                (30, 40 + lane_idx * 40),
                                "%d-lane speed: %d km/h" % (lane_idx + 1, int(v[lane_idx])),
                                color=(0, 0, 255),
                            )

                    mask = np.full_like(frame_gray, 255)
                    for tr in self.tracks:
                        x, y = np.int32(tr[-1])
                        cv.circle(mask, (int(x), int(y)), 3, 0, -1)
                    p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])

                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv.addWeighted(cmask, self.alpha, vis, 1 - self.alpha, 0, vis)

                if writer is not None:
                    writer.write(vis)
                if self.show:
                    cv.imshow("lk_track", vis)
                    if cv.waitKey(1) & 0xFF == ord("q"):
                        break

                ret, frame = self.cam.read()
                if not ret or frame is None:
                    break
        finally:
            if writer is not None:
                writer.release()
            self.cam.release()
            cv.destroyAllWindows()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Vehicle speed estimation from video input.")
    parser.add_argument("video_source", nargs="?", default="0", help="Video source path or camera index.")
    parser.add_argument("--output", default="output.mp4", help="Output video path.")
    parser.add_argument("--show", action="store_true", help="Show preview window.")
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    app = App(args.video_source, output_path=args.output, show=args.show)
    app.run()
    print("Done")


if __name__ == "__main__":
    main()
