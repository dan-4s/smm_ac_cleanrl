"""File to store all functions shared between implementations for SMM-AC testing."""

# Imports
import pyarrow as pa
import pyarrow.parquet as pq

# Constants
STEPS_PER_ENV_ID = {
    "Ant-v4": 1_000_000,
    "HalfCheetah-v4": 1_000_000,
    "Hopper-v4": 1_300_000, # TD3 doesn't seem to converge in time.
    "Humanoid-v4": 3_000_000,
    "Pusher-v4": 1_000_000,
    "Swimmer-v4": 1_000_000,
    "Walker2d-v4": 1_000_000,
}


def get_steps_per_env(env_id: str):
    if(env_id in STEPS_PER_ENV_ID):
        return STEPS_PER_ENV_ID[env_id]
    else:
        print(f"{env_id} not in the list of environments. Setting to 1M.")
        return 1_000_000


def write_and_dump(writer: pq.ParquetWriter, run_data: dict):
    # Only write if there is actually data to avoid schema errors
    if len(run_data["episodic_return"]) > 0:
        table = pa.Table.from_pydict(run_data)
        writer.write_table(table)

        # Clear the dictionary so RAM doesn't grow!
        for key in run_data:
            run_data[key] = []



