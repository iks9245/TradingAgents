"""Import-time compatibility adjustments required only by the graph stack."""

import contextlib
import warnings

# LangChain prepends its own filters during import, so this suppression must be
# installed afterward. Keeping that preload here avoids burdening non-graph
# entry points with an optional LLM dependency.
with contextlib.suppress(ImportError):
    import langchain_core  # noqa: F401

# langgraph-checkpoint 4.0.3 constructs Reviver without allowed_objects during
# import. The upstream fix landed in langgraph#7743 and will make this temporary
# compatibility filter unnecessary after the next checkpoint upgrade.
warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects`.*",
    category=PendingDeprecationWarning,
)
