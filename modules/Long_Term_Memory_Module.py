# Module for handling long-term memory operations using a PostgreSQL database

from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB  # JSONB allows storing Python dictionaries directly in PostgreSQL
from sqlalchemy import Column, Integer, String, DateTime  # Data types for table columns

# Used later to define the table schema
Base = declarative_base()

# Define the structure of the table for storing long-term memory (LTM) data
# This class will later be used to create and update the actual table in the database
# LTM: Long-Term Memory
class LTM_Table_Skeleton(Base):
    __tablename__ = 'ltm_table'
    
    user_id       = Column(String, index=True,  primary_key=True)   # User ID; a unique string
    update_type   = Column(String,  index=True)                      # Type of update: e.g., profile, tasks, procedural memory
    memory_value  = Column(JSONB,   index=True)                      # Memory content as a Python dictionary (stored as JSONB)
    memory_key    = Column(String,  index=True)                      # Unique memory key (typically a UUID)
    creation_time = Column(DateTime(timezone=True), index=True)     # Timestamp of memory creation
    update_time   = Column(DateTime(timezone=True), index=True)     # Timestamp of last memory update

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Database connection string for PostgreSQL
URL_DATABASE = "postgresql://postgres:1538879aA.@localhost:5431/LongTermMemory"

# Create SQLAlchemy engine and session factory
engine = create_engine(URL_DATABASE)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the actual database tables based on the schema defined in Base
Base.metadata.create_all(bind=engine)

from langgraph.store.base import SearchItem
from pydantic import BaseModel, Field
from datetime import datetime

# Pydantic model for data validation and serialization of LTM records
class ltm_pydantic(BaseModel):
    user_id       : str       = Field(description="Unique string identifier for the user.", default=None)
    update_type   : str       = Field(description="Type of memory update: profile, todos, procedural, etc.", default=None)
    memory_value  : dict      = Field(description="Structured memory stored as a dictionary.", default=None)
    memory_key    : str       = Field(description="Unique memory key, usually generated with UUID.", default=None)
    creation_time : datetime  = Field(description="Datetime when the memory was originally created.", default=None)
    update_time   : datetime  = Field(description="Datetime when the memory was last updated.", default=None)

# Converts a LangGraph SearchItem into a validated pydantic LTM object
def langgraph_ltm_to_pydantic_ltm(langgraph_ltm: SearchItem):
    pydantic_ltm = ltm_pydantic(
        user_id       = langgraph_ltm.namespace[1],
        update_type   = langgraph_ltm.namespace[0],
        memory_key    = langgraph_ltm.key,
        memory_value  = langgraph_ltm.value,
        creation_time = langgraph_ltm.created_at,
        update_time   = langgraph_ltm.updated_at
    )
    return pydantic_ltm

# Replaces existing memory in the database with a new memory record
# If a record with the same user_id and memory_key exists, it is deleted before inserting the new one
def replace_ltm_in_db(pydantic_ltm: ltm_pydantic, db: Session):
    
    # Delete existing record (if present)
    db.query(LTM_Table_Skeleton).filter_by(
        user_id    = pydantic_ltm.user_id,
        memory_key = pydantic_ltm.memory_key
    ).delete()

    # Insert the new memory record
    new_record = LTM_Table_Skeleton(
        user_id       = pydantic_ltm.user_id,
        update_type   = pydantic_ltm.update_type,
        memory_value  = pydantic_ltm.memory_value,
        memory_key    = pydantic_ltm.memory_key,
        creation_time = pydantic_ltm.creation_time,
        update_time   = pydantic_ltm.update_time
    )

    db.add(new_record)
    db.commit()
    db.refresh(new_record)

# ------------- OPTIONAL EXAMPLES FOR SERIALIZATION ----------------

# import pickle

# # Saving to pickle (commented out)
# # with open("memory_langgraph.pkl", "wb") as f:
# #     pickle.dump(memory, f)

# # Loading from pickle (commented out)
# with open("memory_langgraph.pkl", "rb") as f:
#     memory = pickle.load(f)

# # Save Pydantic instance to a JSON file (commented out)
# with open("ltm_pydantic_instance.json", "w") as f:
#     f.write(a.model_dump_json())

# # Load JSON into a Pydantic instance (commented out)
# with open("ltm_pydantic_instance.json", "r") as f:
#     data = f.read()
#     b    = ltm_pydantic.model_validate_json(data)
# b