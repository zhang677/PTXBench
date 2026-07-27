"""Sync local traceset directory to HuggingFace Hub.

Uploads the local directory to AccRL/accrl-training dataset repo.
"""

import logging

logger = logging.getLogger(__name__)


def sync_to_hub(
    local_dir: str,
    repo_id: str = "AccRL/accrl-training",
    message: str = "Update",
) -> str:
    """Upload local traceset directory to HuggingFace Hub.

    Returns the commit URL.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    commit_info = api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message,
    )
    url = getattr(commit_info, "commit_url", str(commit_info))
    logger.info("Uploaded %s to %s: %s", local_dir, repo_id, url)
    return url


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Sync traceset to HuggingFace Hub")
    parser.add_argument(
        "--local-dir",
        default="~/accrl-training-data",
        help="Local traceset directory",
    )
    parser.add_argument(
        "--repo-id",
        default="AccRL/accrl-training",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument("--message", default="Update", help="Commit message")
    args = parser.parse_args()

    import os

    local_dir = os.path.expanduser(args.local_dir)
    url = sync_to_hub(local_dir, args.repo_id, args.message)
    print(f"Committed: {url}")
