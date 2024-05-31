import sqlite3
import json
from datetime import date
import uuid
#import pyodbc
#import sqlalchemy as sa
#from sqlalchemy.orm import sessionmaker

# # Database connection details
# DATABASE_CONFIG = {
#     'driver': 'ODBC Driver 18 for SQL Server',
#     'server': 'mysqlserver-caserate.database.windows.net',
#     'database': 'caserate',
#     'username': 'azureuser',
#     'password': 'sVWxpYdhexM!',
# }

# # Construct the database URL
# DATABASE_URL = f"mssql+pyodbc://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}@" \
#                f"{DATABASE_CONFIG['server']}/{DATABASE_CONFIG['database']}?" \
#                f"driver={DATABASE_CONFIG['driver']}"

# # Create an engine
# engine = sa.create_engine(
#     DATABASE_URL,
#     pool_size=10,           # The size of the pool to be maintained
#     max_overflow=20,        # Maximum number of connections to allow in overflow
#     pool_timeout=30,        # Number of seconds to wait for a connection from the pool
#     pool_recycle=1800,      # Number of seconds a connection can be idle before being recycled
# )

# # Create a configured "Session" class
# Session = sessionmaker(bind=engine)

def data_base(data, query):
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('Cases.db')
        curr = conn.cursor()
        # #Driver={ODBC Driver 18 for SQL Server};Server=tcp:mysqlserver-caserate.database.windows.net,1433;Database=caserate;Uid=azureuser;Pwd={your_password_here};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;

        #curr = Session()
    #     session = Session()
    #     tables = session.execute(sa.text("SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'")).fetchall()
        
    #     if not tables:
    #         print("The database is empty: No tables found.")
    #         return True
        
    #     # Check if there is any data in the tables
    #     for table in tables:
    #         table_name = table[0]
    #         row_count = session.execute(sa.text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    #         if row_count > 0:
    #             print(f"The database is not empty: Table '{table_name}' contains data.")
    #             return False
        
    #     print("The database is empty: Tables found but no data in them.")
    #     return True
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     return None
    # finally:
    #     session.close()

        #Insert into Cases table
        client_ID = data["CaseId"]
        current_date = date.today()
        case_description = query
        confidence = data["Confidence(%)"]
        explanation = data["Explanation"]
        PrimaryCaseType = data["PrimaryCaseType"]
        SecondaryCaseType = data["SecondaryCaseType"]
        CaseRating = data["CaseRating"]
        #Case_State = data["Case State"].split()[-1]
        Handling_Firm = data["Handling Firm"]
        try:
            #fetch and map ID from PrimaryCaseTypes
            curr.execute("SELECT PrimaryCaseTypeId FROM PrimaryCaseTypes WHERE CaseType = ?", (PrimaryCaseType,))
            result = curr.fetchone()
            primary_case_type_id = result[0]
            #print("PrimaryCaseTypeId corresponding to", PrimaryCaseType, ":", primary_case_type_id)
        except Exception as error:
            print(error)
        try:
            
            # fetch and map ID from SecondaryCaseTypes
            curr.execute("SELECT ScondaryCaseTypeId FROM SecondaryCaseTypes WHERE CaseType = ?", (SecondaryCaseType,))
            result = curr.fetchone()
            secondary_case_type_id = result[0]
            #print("SecondaryCaseTypeId corresponding to", SecondaryCaseType, ":", secondary_case_type_id)
        except Exception as error:
            print(error)
        
        try:
            Case_State = data["Case State"].split()[-1]
            # fetch and map ID from Caserating
            curr.execute("SELECT CaseRatingId FROM Caserating WHERE CastRating = ?", (CaseRating,))
            result = curr.fetchone()
            case_rating_id = result[0]
            #print("CaseRatingId corresponding to", CaseRating, ":", case_rating_id)
        except Exception as error:
            print(error)

        try:
            # fetch and map ID from CaseStates
            curr.execute("SELECT CaseStateId FROM CaseStates WHERE Name = ?", (Case_State,))
            result = curr.fetchone()
            case_state_id = result[0]
            #print("CaseStateId corresponding to", Case_State, ":", case_state_id)
        except Exception as error:
            print(error)
        
        try:
            # fetch and map ID from HandlingFirms
            curr.execute("SELECT HandlingFirmId FROM HandlingFirms WHERE Name = ?", (Handling_Firm,))
            result = curr.fetchone()
            handling_firm_id = result[0]
            #print("HandlingFirmId corresponding to", Handling_Firm, ":", handling_firm_id)
        except Exception as error:
            print(error)
            
        curr.execute("INSERT INTO Cases (CaseId, Date, CaseDescription, PrimaryCaseTypeId, SecondaryCaseTypeId, CaseRatingId, CaseStateId, HandlingFirmId, Confidence, Explanation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (client_ID, current_date, case_description, primary_case_type_id, secondary_case_type_id, case_rating_id, case_state_id, handling_firm_id, confidence, explanation))  # Assuming the PrimaryCaseTypeId, SecondaryCaseTypeId, CaseRatingId, CaseStateId, HandlingFirmId are 1 for simplicity

        # Fetch and print data from the table
        curr.execute("SELECT * FROM Cases")
        #print("Table Data:")
        rows = curr.fetchall()
        for row in rows:
            print(row, end="\n\n")

        #curr.close()
        # Commit changes and close connection
        curr.commit()
        curr.close()
    except Exception as error:
        print(error)
        
# if __name__ == "__main__":
#     data = {"PrimaryCaseType": "General Injury", "SecondaryCaseType": "Automobile Accident", "CaseRating": "Tier 2", "Case State": "Unknown", "Is Workers Compensation (Yes/No)?": "No", "Confidence(%)": "85%", "Explanation": "Based on the description provided, the client was in a car accident and sustained injuries to the face, neck, back, and hip. The injuries mentioned fall under the Tier 2 category for General Injury cases involving Automobile Accidents. The case state is provided as 'Unknown' as there is no mention of the state in the description. The case does not involve a workplace incident, therefore the 'Is Workers Compensation' is 'No'. The confidence level is 85% as the case details match the criteria for General Injury and Automobile Accident case types, however, the case state is uncertain due to the lack of information in the description.", "Handling Firm": "SAD"}
#     query = "Mr. Sabir has a concussion and back injury as result of an accident he suffered with Uber. He would like assistance with his auto accident claim."
#     generated_uuid = uuid.uuid4()
#     data_base(data, query)

