import requests

def api_fetch(api_url, query, access_token):
    """
    Function to make a POST request to the API endpoint.

    Parameters:
        - api_url (str): The URL of the API endpoint.
        - input_data (dict): The input data to be sent to the API endpoint.
        - access_token (str): The access token for authorization.

    Returns:
        - dict: The JSON response from the API.
    """

    input_data = {
        "msg": query
    }

    # Prepare headers with the access token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    try:
        # Make POST request to the API endpoint
        response = requests.post(api_url, headers=headers, json=input_data)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Return the JSON response
            return response.json()
        else:
            # Print error message if request was unsuccessful
            print(f"Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        # Print error message if an exception occurs
        print(f"Error: {str(e)}")
        return None

# Example usage:
api_url = "http://20.84.101.4:8080/case-classifier"

query = "Kevin Johnson DOI 9/28/2023 Swift Transportation Phila PA Injury back damage CL was working driving an 18 wheeler and was hit and injured by another commercial truck. CL injured his back and had back surgery that was diagnosed as L4 L5 damage. CL lost time form work and loss of income. Still seeing doctors and taking pain meds. Damages CL uses cane and suffers from limited mobility"
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTcxMjgyMzczMX0.GGhi_xlnnVxl9oeRF4x8uixuDYSpZfCYudklfwHVx0c"

response = api_fetch(api_url, query, access_token)
print(response)


