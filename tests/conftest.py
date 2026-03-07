from __future__ import annotations

import os
import tempfile
from pathlib import Path


_TEST_ROOT = Path(tempfile.mkdtemp(prefix="multishell-tests-"))
os.environ.setdefault("MULTISHELL_STATE_ROOT", str(_TEST_ROOT / ".multishell"))
os.environ.setdefault("MULTISHELL_ENV_FILE", str(_TEST_ROOT / ".env"))
