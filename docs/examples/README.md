# Example input formats

`route_template.csv` shows the **format** of a GPS route file for video detection. The coordinates are a
hand-written example (not a recorded drive) — replace them with a real track exported from a phone GPS
logger or dash cam that was recording while the video was filmed.

* `time_s` — seconds since the first frame of the video
* `lat`, `lon` — WGS84 decimal degrees

Alternatives accepted by the API: a CSV with an ISO-8601 `timestamp` column instead of `time_s`
(the first row is taken as the video start), or a GPX 1.1 track whose points have `<time>` elements.
