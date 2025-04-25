import sqlite3
import json
from typing import List, Dict, Optional
import os
from config import DATABASE_FILE, PERSONAS_DIR

# Ensure personas directory exists
os.makedirs(PERSONAS_DIR, exist_ok=True)

def init_db():
    """Initializes the SQLite database and creates the personas table if it doesn't exist."""
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS personas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    role TEXT,
                    bio TEXT,
                    psychological_traits TEXT, -- Stored as JSON string
                    influences TEXT,           -- Stored as JSON string
                    biases TEXT,               -- Stored as JSON string
                    historical_behavior TEXT,
                    tone TEXT,
                    goals TEXT,                -- Stored as JSON string
                    expected_behavior TEXT
                )
            ''')
            conn.commit()
        print("Database initialized successfully.")
    except sqlite3.Error as e:
        print(f"Database error during initialization: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during DB initialization: {e}")


def _dict_factory(cursor, row):
    """Converts database rows into dictionaries."""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    # Deserialize JSON fields
    for field in ['psychological_traits', 'influences', 'biases', 'goals']:
        if d.get(field) and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON for field '{field}' in persona ID {d.get('id')}")
                d[field] = [] # Default to empty list on error
    return d

def execute_query(query: str, params: tuple = (), fetch_one: bool = False, commit: bool = False):
    """Executes a generic SQL query."""
    result = None
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            conn.row_factory = _dict_factory # Use dictionary factory for results
            cursor = conn.cursor()
            cursor.execute(query, params)
            if fetch_one:
                result = cursor.fetchone()
            elif not commit: # Don't fetchall if it's an INSERT/UPDATE/DELETE unless needed
                 result = cursor.fetchall()

            if commit:
                conn.commit()
    except sqlite3.IntegrityError as e:
         print(f"Database Integrity Error: {e}. Likely duplicate name.")
         # Optionally re-raise or return a specific error code/message
         raise # Re-raise for handling in the calling function
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        # Optionally re-raise or handle more gracefully
        raise
    except Exception as e:
        print(f"An unexpected error occurred during query execution: {e}")
        raise
    return result


def save_persona(persona: Dict) -> Optional[int]:
    """Saves or updates a persona in the database based on name."""
    query_select = "SELECT id FROM personas WHERE name = ?"
    query_insert = """
        INSERT INTO personas (name, role, bio, psychological_traits, influences, biases, historical_behavior, tone, goals, expected_behavior)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    query_update = """
        UPDATE personas SET role=?, bio=?, psychological_traits=?, influences=?, biases=?, historical_behavior=?, tone=?, goals=?, expected_behavior=?
        WHERE name=?
    """
    params = (
        persona.get('name'),
        persona.get('role'),
        persona.get('bio'),
        json.dumps(persona.get('psychological_traits', [])),
        json.dumps(persona.get('influences', [])),
        json.dumps(persona.get('biases', [])),
        persona.get('historical_behavior'),
        persona.get('tone'),
        json.dumps(persona.get('goals', [])),
        persona.get('expected_behavior')
    )

    try:
        existing = execute_query(query_select, (persona.get('name'),), fetch_one=True)
        if existing:
            # Update existing persona
            update_params = params[1:] + (persona.get('name'),) # role onwards, plus name for WHERE
            execute_query(query_update, update_params, commit=True)
            print(f"Persona '{persona.get('name')}' updated in database.")
            return existing['id']
        else:
            # Insert new persona
            execute_query(query_insert, params, commit=True)
            # Get the last inserted ID (less reliable without proper cursor handling)
            # It's better to fetch the persona again if ID is crucial immediately
            print(f"Persona '{persona.get('name')}' saved to database.")
            # Fetch the newly inserted persona to get its ID
            new_persona = execute_query(query_select, (persona.get('name'),), fetch_one=True)
            return new_persona['id'] if new_persona else None
    except sqlite3.Error as e:
        print(f"Error saving persona '{persona.get('name')}': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error saving persona: {e}")
        return None


def get_all_personas() -> List[Dict]:
    """Retrieves all personas from the database."""
    query = "SELECT * FROM personas ORDER BY name"
    try:
        personas = execute_query(query)
        return personas if personas else []
    except Exception as e:
        print(f"Error getting all personas: {e}")
        return []

def get_persona_by_name(name: str) -> Optional[Dict]:
    """Retrieves a single persona by name."""
    query = "SELECT * FROM personas WHERE name = ?"
    try:
        persona = execute_query(query, (name,), fetch_one=True)
        return persona
    except Exception as e:
        print(f"Error getting persona by name '{name}': {e}")
        return None

def update_persona(persona: Dict) -> bool:
    """Updates an existing persona in the database using its ID."""
    if 'id' not in persona:
        print("Error: Cannot update persona without an ID.")
        return False

    query = """
        UPDATE personas SET
            name=?, role=?, bio=?, psychological_traits=?, influences=?, biases=?,
            historical_behavior=?, tone=?, goals=?, expected_behavior=?
        WHERE id=?
    """
    params = (
        persona.get('name'),
        persona.get('role'),
        persona.get('bio'),
        json.dumps(persona.get('psychological_traits', [])),
        json.dumps(persona.get('influences', [])),
        json.dumps(persona.get('biases', [])),
        persona.get('historical_behavior'),
        persona.get('tone'),
        json.dumps(persona.get('goals', [])),
        persona.get('expected_behavior'),
        persona.get('id')
    )
    try:
        execute_query(query, params, commit=True)
        print(f"Persona ID {persona.get('id')} ({persona.get('name')}) updated successfully.")
        return True
    except Exception as e:
        print(f"Error updating persona ID {persona.get('id')}: {e}")
        return False


def delete_persona(persona_id: int) -> bool:
    """Deletes a persona from the database by ID."""
    query = "DELETE FROM personas WHERE id = ?"
    try:
        execute_query(query, (persona_id,), commit=True)
        print(f"Persona ID {persona_id} deleted successfully.")
        return True
    except Exception as e:
        print(f"Error deleting persona ID {persona_id}: {e}")
        return False

# Example usage (optional, for testing)
if __name__ == "__main__":
    print("Initializing DB...")
    init_db()
    print("\nAttempting to save/update a test persona...")
    test_persona = {
        "name": "Test User Alpha",
        "role": "Tester",
        "bio": "A persona created for testing purposes.",
        "psychological_traits": ["curious", "methodical"],
        "influences": ["test cases", "requirements"],
        "biases": ["automation bias"],
        "historical_behavior": "Follows test scripts",
        "tone": "neutral",
        "goals": ["find bugs", "verify functionality"],
        "expected_behavior": "Executes tests and reports results objectively."
    }
    save_persona(test_persona)

    print("\nFetching all personas...")
    all_p = get_all_personas()
    print(f"Found {len(all_p)} personas.")
    # for p in all_p:
    #     print(f"- {p.get('name')} (ID: {p.get('id')})")

    print("\nFetching 'Test User Alpha' by name...")
    fetched = get_persona_by_name("Test User Alpha")
    if fetched:
        print(f"Found: {fetched.get('name')}, Role: {fetched.get('role')}")
        fetched['role'] = "Senior Tester"
        fetched['goals'] = ["find critical bugs", "improve test coverage"]
        print("\nUpdating 'Test User Alpha'...")
        update_persona(fetched)
        updated = get_persona_by_name("Test User Alpha")
        print(f"Updated Role: {updated.get('role')}, Updated Goals: {updated.get('goals')}")

        # print("\nDeleting 'Test User Alpha'...")
        # delete_persona(fetched['id'])
        # print("Attempting to fetch deleted user...")
        # deleted = get_persona_by_name("Test User Alpha")
        # print(f"Found after delete: {deleted}")
    else:
        print("Test User Alpha not found.")
