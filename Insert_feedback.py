import sqlite3
import json
import uuid

def feedback_table(data, generated_uuid):
    # Insert into Feedback table
    try:
        conn = sqlite3.connect('Cases.db')
        curr = conn.cursor()
        client_ID = data["CaseId"]
        FeedbackId = str(generated_uuid)
        primary_case_type = data["PrimaryCaseType"]
        secondary_case_type = data["SecondaryCaseType"]
        case_rating = data["CaseRating"]
        # case_state = data["CaseState"].split()[-1]
        handling_firm = data["HandlingFirm"]
    
        try:
            #fetch and map ID from PrimaryCaseTypes
            curr.execute("SELECT PrimaryCaseTypeId FROM PrimaryCaseTypes WHERE CaseType = ?", (primary_case_type,))
            result = curr.fetchone()
            primary_case_type_id = result[0]
            #print("PrimaryCaseTypeId corresponding to", primary_case_type, ":", primary_case_type_id)

        except Exception as error:
            print(error)

        try:
            # fetch and map ID from SecondaryCaseTypes
            curr.execute("SELECT ScondaryCaseTypeId FROM SecondaryCaseTypes WHERE CaseType = ?", (secondary_case_type,))
            result = curr.fetchone()
            secondary_case_type_id = result[0]
            #print("SecondaryCaseTypeId corresponding to", secondary_case_type, ":", secondary_case_type_id)
        except Exception as error:
            print(error)

        try:
            # fetch and map ID from Caserating
            curr.execute("SELECT CaseRatingId FROM Caserating WHERE CastRating = ?", (case_rating,))
            result = curr.fetchone()
            case_rating_id = result[0]
            #print("CaseRatingId corresponding to", case_rating, ":", case_rating_id)
        except Exception as error:
            print(error)

        try:
            case_state = data["CaseState"].split()[-1]
            # fetch and map ID from CaseStates
            curr.execute("SELECT CaseStateId FROM CaseStates WHERE Name = ?", (case_state,))
            result = curr.fetchone()
            case_state_id = result[0]
            #print("CaseStateId corresponding to", case_state, ":", case_state_id)
        except Exception as error:
            print(error)
        try:
            # fetch and map ID from CaseStates
            curr.execute("SELECT HandlingFirmId FROM HandlingFirms WHERE Name = ?", (handling_firm,))
            result = curr.fetchone()
            handling_firm_id = result[0]
            #print("HandlingFirmId corresponding to", handling_firm, ":", handling_firm_id)
        except Exception as error:
            print(error)
        
        # Update the FeedbackId in the Cases table
        curr.execute("UPDATE Cases SET FeedbackId = ? WHERE CaseId = ?", (FeedbackId, client_ID))

        curr.execute("INSERT INTO Feedback (FeedbackId, PrimaryCaseTypeId, SecondaryCaseTypeId, CaseRatingId, CaseStateId, HandlingFirmId) VALUES (?, ?, ?, ?, ?, ?)",
                    (FeedbackId, primary_case_type_id, secondary_case_type_id, case_rating_id, case_state_id, handling_firm_id))

        # Fetch and print data from the Cases table
        curr.execute("SELECT * FROM Cases")
        print("Cases Table Data:")
        rows = curr.fetchall()
        print(rows, end="\n\n")

        # Fetch and print data from the table
        curr.execute("SELECT * FROM Feedback")
        print("Table Data:")
        rows = curr.fetchall()
        print(rows)
        
        curr.close()
        # Commit changes and close connection
        conn.commit()
        conn.close()
    except Exception as error:
        print(error)

# if __name__ == "__main__":
#     data = {"CaseDescription": "Kevin Johnson DOI 9/28/2023 Swift Transportation Phila PA Injury back damage CL was working driving an 18 wheeler and was hit and injured by another commercial truck. CL injured his back and had back surgery that was diagnosed as L4 L5 damage. CL lost time form work and loss of income. Still seeing doctors and taking pain meds. Damages CL uses cane and suffers from limited mobility", "PrimaryCaseType": "Workers Compensation", "SecondaryCaseType": "Automobile Accident", "CaseRating": "Tier 3", "CaseState": "PA Pennsylvania", "IsWorkersCompensation": "Yes", "Confidence(%)": "90%", "Explanation": "The description indicates that the client was driving an 18 wheeler for work when he was hit and injured by another commercial truck. This falls under the Workers Compensation primary case type. The fact that the injury was caused by an automobile accident makes the secondary case type Automobile Accident. The injury is described as L4 L5 damage, which requires surgery, putting it in Tier 4. The incident happened in Philadelphia, PA, so the case state is PA Pennsylvania. The description also specifies that the client was working at the time, so this is a Workers Compensation case. The confidence level is high due to the clear description of the incident and the identification of the state.\n\nThe case rating is Tier 3 and case types are Workers Compensation and Automobile Accident, which fall under the 'Any' category. Therefore, according to the handling rules for the state, the recommended handling firm is 'REFER TO PA SECTION'.", "HandlingFirm": "REFER TO PA SECTION", "CaseId": "2b64a078-a54e-430d-a226-a05f21448723"}
#     query = "Mr. Sabir has a concussion and back injury as result of an accident he suffered with Uber. He would like assistance with his auto accident claim."
#     generated_uuid = uuid.uuid4()
    
#     feedback_table(data, generated_uuid)
