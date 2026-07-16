# Sarween-Python
Camera Vision App for TTRPGs

## Tracking Regression Tests

You can run the combo tracker against a recorded video without opening the
camera/control-panel UI:

```bash
python3 tracking_regression.py run sarween_rec_20260420_163421.mp4
```

To turn video behavior into assertions, edit the starter
`tests/fixtures/tracking_cases.json` using
`tests/fixtures/tracking_cases.example.json` as the shape, then run:

```bash
bash scripts/run_tracking_regression.sh
```

If your default `python3` does not have OpenCV installed, point the helper at a
Python environment that does:

```bash
PYTHON_BIN=/path/to/python-with-cv2 bash scripts/run_tracking_regression.sh
```

Each expectation can pin a mini, source cell, destination cell, expected video
timestamp, and tolerance. By default, unexpected emitted moves fail the case so
false positives are caught as well as missed real movements.

