"""
Account tools — used by AccountAgent.

These tools query the DB for user-specific data.
Mock data is acceptable for the take-home; the integration matters.

TODO for candidate: implement these tools.
"""
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str  # passed | failed | cancelled
    branch: str
    started_at: datetime
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


async def get_recent_builds(user_id: str, limit: int = 5) -> list[BuildSummary]:
    """
    Return the most recent builds for a user, newest first.

    For the take-home: returning mock/seeded data is fine.
    The key evaluation point is that this is wired as an ADK tool
    and the agent correctly invokes it when the user asks about builds.
    """
    normalized_limit = max(1, min(limit, 20))
    seed = _stable_int(user_id)

    pipelines = ["core", "deploy", "security", "release"]
    branches = ["main", "develop", "release", "hotfix"]
    statuses = ["passed", "failed", "cancelled"]

    now = datetime.utcnow()
    builds: list[BuildSummary] = []
    for index in range(normalized_limit):
        pipeline = pipelines[(seed + index) % len(pipelines)]
        branch = branches[(seed // 3 + index) % len(branches)]
        status = statuses[(seed // 7 + index) % len(statuses)]
        duration_seconds = 120 + ((seed + index * 13) % 900)
        started_at = now - timedelta(hours=index)

        builds.append(
            BuildSummary(
                build_id=f"bld_{seed:08x}_{index + 1}",
                pipeline=pipeline,
                status=status,
                branch=branch,
                started_at=started_at,
                duration_seconds=duration_seconds,
            )
        )

    return builds


async def get_account_status(user_id: str) -> AccountStatus:
    """
    Return current account status (plan, usage limits).

    For the take-home: mock data is fine.
    """
    plan_tier = _plan_tier_from_user_id(user_id)
    seed = _stable_int(user_id)

    if plan_tier == "enterprise":
        concurrent_builds_limit = 50
        storage_limit_gb = 500.0
    elif plan_tier == "pro":
        concurrent_builds_limit = 10
        storage_limit_gb = 100.0
    else:
        concurrent_builds_limit = 2
        storage_limit_gb = 10.0

    concurrent_builds_used = (seed % concurrent_builds_limit) + 1
    storage_used_gb = round((seed % int(storage_limit_gb * 10)) / 10.0, 1)

    return AccountStatus(
        user_id=user_id,
        plan_tier=plan_tier,
        concurrent_builds_used=concurrent_builds_used,
        concurrent_builds_limit=concurrent_builds_limit,
        storage_used_gb=storage_used_gb,
        storage_limit_gb=storage_limit_gb,
    )


def _stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode()).hexdigest()
    return int(digest[:8], 16)


def _plan_tier_from_user_id(user_id: str) -> str:
    lowered = user_id.lower()
    if "enterprise" in lowered:
        return "enterprise"
    if "pro" in lowered:
        return "pro"
    return "free"
