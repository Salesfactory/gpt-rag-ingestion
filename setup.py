import logging
import time
import os
import requests
from typing import Optional
import argparse
import json
from azure.mgmt.web import WebSiteManagementClient
from azure.identity import (
    ManagedIdentityCredential,
    AzureCliCredential,
    ChainedTokenCredential,
)
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

# Set up logging configuration globally
logging.getLogger("azure").setLevel(logging.WARNING)

from azure.identity import ManagedIdentityCredential, AzureCliCredential, ChainedTokenCredential
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
# Set up logging configuration globally
logging.getLogger('azure').setLevel(logging.WARNING)

def call_search_api(search_service, search_api_version, resource_type, resource_name, method, credential, body=None):
    """
    Calls the Azure Search API with the specified parameters.

    Args:
        search_service (str): The name of the Azure Search service.
        search_api_version (str): The version of the Azure Search API to use.
        resource_type (str): The type of resource to access (e.g. "indexes", "docs").
        resource_name (str): The name of the resource to access.
        method (str): The HTTP method to use (either "get" or "put").
        credential (TokenCredential): An instance of a TokenCredential class that can provide an access token.
        body (dict, optional): The JSON payload to include in the request body (for "put" requests).

    Returns:
        None

    Raises:
        ValueError: If the specified HTTP method is not "get" or "put".
        HTTPError: If the response status code is 400 or greater.

    """    
    # get the token
    token = credential.get_token("https://search.azure.com/.default").token
    headers = {
        "Authorization": f"Bearer {token}",
        'Content-Type': 'application/json'
        # 'api-key': SEARCH_API_KEY
    }
    search_endpoint = f"https://{search_service}.search.windows.net/{resource_type}/{resource_name}?api-version={search_api_version}"
    response = None
    try:
        if method not in ["get", "put", "delete"]:
            logging.warning(f"[call_search_api] Invalid method {method} ")

        # get and put processing
        if method == "get":
            response = requests.get(search_endpoint, headers=headers)
        elif method == "put":
            response = requests.put(search_endpoint, headers=headers, json=body)

        # delete processing
        if method == "delete":
            response = requests.delete(search_endpoint, headers=headers)
            status_code = response.status_code
            logging.info(f"[call_search_api] Successfully called search API {method} {resource_type} {resource_name}. Code: {status_code}.")

        if response is not None:
            status_code = response.status_code
            if status_code >= 400:
                logging.warning(f"[call_search_api] {status_code} code when calling search API {method} {resource_type} {resource_name}. Reason: {response.reason}.")
                try:
                    response_text_dict = json.loads(response.text)
                    logging.warning(f"[call_search_api] {status_code} code when calling search API {method} {resource_type} {resource_name}. Message: {response_text_dict['error']['message']}")        
                except json.JSONDecodeError:
                    logging.warning(f"[call_search_api] {status_code} Response is not valid JSON. Raw response:\n{response.text}")
        
            else:
                logging.info(f"[call_search_api] Successfully called search API {method} {resource_type} {resource_name}. Code: {status_code}.")


    except Exception as e:
        error_message = str(e)
        logging.error(f"Error when calling search API {method} {resource_type} {resource_name}. Error: {error_message}")

def get_function_key(subscription_id, resource_group, function_app_name, credential):
    """
    Returns an API key for the given function.

    Parameters:
    subscription_id (str): The subscription ID.
    resource_group (str): The resource group name.
    function_app_name (str): The name of the function app.
    credential (str): The credential to use.

    Returns:
    str: A unique key for the function.
    """    
    logging.info(f"Obtaining function key after creating or updating its value.")
    accessToken = f"Bearer {credential.get_token('https://management.azure.com/.default').token}"
    # Get key
    requestUrl = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Web/sites/{function_app_name}/functions/document_chunking/keys/mykey?api-version=2022-03-01"
    
    requestHeaders = {
        "Authorization": accessToken,
        "Content-Type": "application/json"
    }
    data = {
        "properties": {
            "name": "mykey"  # Omit the 'value' field to let Azure generate the key
        }
    }

    response = requests.put(requestUrl, headers=requestHeaders, json=data)
    response_json = json.loads(response.content.decode('utf-8'))
    # print(response_json)
    try:
        function_key = response_json['properties']['value']
    except Exception as e:
        function_key = None
        logging.error(f"Error when getting function key. Details: {str(e)}.")        
    return function_key

def approve_private_link_connections(access_token, subscription_id, resource_group, service_name, service_type, api_version):
    """
    Approves private link service connections for a given service using
    the "GET-then-PUT" pattern to ensure all required fields are present.

    Args:
        access_token (str): The access token used for authorization.
        subscription_id (str): The subscription ID.
        resource_group (str): The resource group name.
        service_name (str): The name of the service.
        service_type (str): The type of the service (e.g., 'Microsoft.Storage/storageAccounts').
        api_version (str): The API version.

    Returns:
        None

    Note:
        Instead of raising an exception on errors, we log a warning.
        This updated version performs a "GET-then-PUT" for each connection
        to avoid 'InvalidValuesForRequestParameters' errors.
    """
    logging.info(f"[approve_private_link_connections] Access token: {access_token[:10]}...")
    logging.info(f"[approve_private_link_connections] Subscription ID: {subscription_id}")
    logging.info(f"[approve_private_link_connections] Resource group: {resource_group}")
    logging.info(f"[approve_private_link_connections] Service name: {service_name}")
    logging.info(f"[approve_private_link_connections] Service type: {service_type}")
    logging.info(f"[approve_private_link_connections] API version: {api_version}")

    # List all private endpoint connections for the given resource
    list_url = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}/providers/{service_type}/{service_name}"
        f"/privateEndpointConnections?api-version={api_version}"
    )
    logging.debug(f"[approve_private_link_connections] Request URL: {list_url}")

    request_headers = {
        "Authorization": access_token,
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(list_url, headers=request_headers)
        response.raise_for_status()
        response_json = response.json()

        if 'value' not in response_json:
            logging.error(
                f"Unexpected response structure when fetching private link connections. "
                f"Response content: {response.content}"
            )
            return  # No connections to approve or error in structure

        # Iterate over all connections in the array
        for connection in response_json["value"]:
            connection_id = connection["id"]  # Full ARM ID
            connection_name = connection["name"]
            status = connection["properties"]["privateLinkServiceConnectionState"]["status"]
            logging.info(f"[approve_private_link_connections] Checking connection '{connection_name}'. Status: {status}.")

            # Approve only if status is 'Pending' 
            if status.lower()== "pending":
                # 1) GET the entire connection resource so we can PUT it back intact
                single_connection_url = f"https://management.azure.com{connection_id}?api-version={api_version}"
                logging.debug(f"[approve_private_link_connections] GET single connection URL: {single_connection_url}")
                try:
                    single_conn_response = requests.get(single_connection_url, headers=request_headers)
                    single_conn_response.raise_for_status()
                    full_conn_resource = single_conn_response.json()
                except requests.HTTPError as http_err:
                    logging.warning(
                        f"Failed to GET full connection resource for '{connection_name}': {http_err}. "
                        f"Response: {single_conn_response.text if 'single_conn_response' in locals() else ''}"
                    )
                    continue

                # 2) Update the status to "Approved" within the retrieved resource
                full_conn_resource["properties"]["privateLinkServiceConnectionState"]["status"] = "Approved"
                full_conn_resource["properties"]["privateLinkServiceConnectionState"]["description"] = "Approved by setup script"

                # 3) PUT the entire resource (with updated status)
                logging.debug(f"[approve_private_link_connections] PUT single connection URL: {single_connection_url}")
                approve_response = requests.put(single_connection_url, headers=request_headers, json=full_conn_resource)

                if approve_response.status_code in [200, 202]:
                    logging.info(
                        f"Approved private endpoint connection '{connection_name}' for service '{service_name}'."
                    )
                else:
                    logging.warning(
                        f"Warning: Failed to approve private endpoint connection '{connection_name}' "
                        f"for service '{service_name}'. Status Code: {approve_response.status_code}, "
                        f"Response: {approve_response.text}"
                    )
            elif status.lower() == "approved":
                logging.info(f"[approve_private_link_connections] Connection '{connection_name}' is already Approved. Skipping re-approval.")
                continue
            
    except requests.HTTPError as http_err:
        logging.warning(
            f"HTTP error occurred when listing/approving private link connections: {http_err}. "
            f"Response: {response.text}"
        )
    except Exception as e:
        logging.warning(f"Error occurred when approving private link connections: {e}")

def approve_search_shared_private_access(subscription_id, resource_group, storage_resource_group, aoai_resource_group, function_app_name, storage_account_name, openai_service_name, credential):
    """
    Approves Shared Private Access requests for private endpoints for AI Search, storage account, function app, and Azure OpenAI Service.

    Args:
        subscription_id (str): The subscription ID.
        resource_group (str): The resource group name.
        function_app_name (str): The name of the function app.
        storage_account_name (str): The name of the storage account.
        openai_service_name (str): The name of the Azure OpenAI service.

    Returns:
        None

    Raises:
        Exception: If approval fails.
    """ 
    try:
        logging.info("Approving Shared Private Access requests for storage, function app, and Azure OpenAI Service if needed.")
        
        # Obtain the access token
        try:
            token_response = credential.get_token("https://management.azure.com/.default")
            access_token = f"Bearer {token_response.token}"
            logging.info("Obtained access token successfully.")
        except ClientAuthenticationError as e:
            logging.error(f"Authentication failed when obtaining access token: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error when obtaining access token: {e}")
            raise

        # Approve private link connection for storage account
        try:
            approve_private_link_connections(
                access_token, 
                subscription_id, 
                storage_resource_group, 
                storage_account_name, 
                'Microsoft.Storage/storageAccounts', 
                '2023-01-01'
            )
            logging.info(f"[approve_private_link_connections] Approved private link connections for Storage Account: {storage_account_name}.")
        except Exception as e:
            logging.error(f"Failed to approve private link connections for Storage Account '{storage_account_name}': {e}")
            raise
        
        # Approve private link connection for function app
        try:
            approve_private_link_connections(
                access_token, 
                subscription_id, 
                resource_group, 
                function_app_name, 
                'Microsoft.Web/sites', 
                '2022-09-01'
            )
            logging.info(f"[approve_private_link_connections] Approved private link connections for Function App: {function_app_name}.")
        except Exception as e:
            logging.error(f"Failed to approve private link connections for Function App '{function_app_name}': {e}")
            raise

        # Approve private link connection for Azure OpenAI Service
        try:
            approve_private_link_connections(
                access_token, 
                subscription_id, 
                aoai_resource_group, 
                openai_service_name, 
                'Microsoft.CognitiveServices/accounts', 
                '2022-10-01'
            )
            logging.info(f"Approved private link connections for Azure OpenAI Service: {openai_service_name}.")
        except Exception as e:
            logging.error(f"Failed to approve private link connections for Azure OpenAI Service '{openai_service_name}': {e}")
            raise
    
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error when approving private link service connection. Please do it manually. Error: {error_message}")
        raise

def create_indexer_body(indexer_name: str, index_name: str = None):
            # Azure Cognitive Search settings from .env
        service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
        api_version = "2024-11-01-preview" 
        admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        indexer_name = f"nam-indexer-chunk-documents"
        # Endpoint URL
        endpoint = f"https://{service_name}.search.windows.net/indexers/{indexer_name}?api-version={api_version}"
        
        # Headers
        headers = {
            'Content-Type': 'application/json',
            'api-key': admin_key
        }
        body = {
        "name": indexer_name,
        "dataSourceName": f"{index_name}-datasource",
        "targetIndexName": f"{index_name}",
        "skillsetName": f"{index_name}-skillset-chunking",
        "schedule": {"interval": "P1D"},
        "fieldMappings": [
            {
                "sourceFieldName": "metadata_storage_path",
                "targetFieldName": "id",
                "mappingFunction": {
                    "name": "base64Encode"
                }
            },
            {
                "sourceFieldName": "organization_id",
                "targetFieldName": "organization_id",
                "mappingFunction": None
            }
        ],
        "outputFieldMappings": [],
        "parameters": {
            "batchSize": 1,
            "maxFailedItems": -1,
            "maxFailedItemsPerBatch": -1,
            "configuration": {
                "dataToExtract": "contentAndMetadata",
                "parsingMode": "default"
            }
        }
    }
        
        # First, try to delete the existing indexer if it exists
        try:
            delete_response = requests.delete(endpoint, headers=headers)
            print(f"Delete existing indexer response: {delete_response.status_code}")
        except Exception as e:
            print(f"Error deleting existing indexer: {e}")
        
        # Create the new indexer
        try:
            response = requests.put(endpoint, headers=headers, json=body)
            
            if response.status_code in [200, 201]:
                print("Indexer created successfully!")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"Error creating indexer. Status code: {response.status_code}")
                print(f"Error message: {response.text}")
                
        except Exception as e:
            print(f"Error creating indexer: {e}")
def execute_setup(subscription_id, resource_group, function_app_name, search_principal_id, azure_search_use_mis, enable_managed_identities, enable_env_credentials):
    """
    This function performs the necessary steps to set up the ingestion sub components, such as creating the required datastores and indexers.
    
    Args:
        subscription_id (str): The subscription ID of the Azure subscription to use.
        resource_group (str): The name of the resource group containing the solution resources.
        function_app_name (str): The name of the function app to use.
        search_principal_id (str): The principal ID of the search managed identity.
        azure_search_use_mis (bool): Whether to use Search Service Managed Identity to Connect to data ingestion function
        enable_managed_identities (bool, optional): Whether to use VM's managed identities to run the setup, defaults to False.
        enable_env_credentials (bool): Whether to use environment credentials to run the setup.

    Returns:
        None
    """    
    logging.info(f"Getting function app {function_app_name} properties.") 
    credential = ChainedTokenCredential(
        ManagedIdentityCredential(),
        AzureCliCredential()
    )
    web_mgmt_client = WebSiteManagementClient(credential, subscription_id)
    function_app_settings = web_mgmt_client.web_apps.list_application_settings(resource_group, function_app_name)
    function_endpoint = f"https://{function_app_name}.azurewebsites.net"
    azure_openai_service_name = function_app_settings.properties["AZURE_OPENAI_SERVICE_NAME"]
    search_service = function_app_settings.properties["AZURE_SEARCH_SERVICE"]
    search_analyzer_name= function_app_settings.properties["SEARCH_ANALYZER_NAME"]
    search_api_version = function_app_settings.properties.get("SEARCH_API_VERSION", "2024-07-01") 
    search_index_interval = function_app_settings.properties["SEARCH_INDEX_INTERVAL"]
    search_index_name = function_app_settings.properties["SEARCH_INDEX_NAME"]
    storage_container = function_app_settings.properties["STORAGE_CONTAINER"]
    storage_account_name = function_app_settings.properties["STORAGE_ACCOUNT_NAME"]
    network_isolation = True if function_app_settings.properties["NETWORK_ISOLATION"].lower() == "true" else False
    storage_container = function_app_settings.properties["STORAGE_CONTAINER"]
    storage_account_name = function_app_settings.properties["STORAGE_ACCOUNT_NAME"]
    azure_openai_embedding_deployment = function_app_settings.properties.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding")
    azure_openai_embedding_model = function_app_settings.properties.get("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    azure_embeddings_vector_size = function_app_settings.properties.get("AZURE_EMBEDDINGS_VECTOR_SIZE", "3072")
    azure_storage_resource_group = function_app_settings.properties["AZURE_STORAGE_ACCOUNT_RG"]
    azure_aoai_resource_group = function_app_settings.properties["AZURE_AOAI_RG"]    

    logging.info(f"[execute_setup] Function endpoint: {function_endpoint}")
    logging.info(f"[execute_setup] Search service: {search_service}")
    logging.info(f"[execute_setup] Search analyzer name: {search_analyzer_name}")
    logging.info(f"[execute_setup] Search API version: {search_api_version}")
    logging.info(f"[execute_setup] Search index interval: {search_index_interval}")
    logging.info(f"[execute_setup] Search index name: {search_index_name}")
    logging.info(f"[execute_setup] Storage container: {storage_container}")
    logging.info(f"[execute_setup] Storage account name: {storage_account_name}")
    logging.info(f"[execute_setup] Embedding deployment name: {azure_openai_embedding_deployment}")
    logging.info(f"[execute_setup] Embedding model: {azure_openai_embedding_model}")
    logging.info(f"[execute_setup] Embedding vector size: {azure_embeddings_vector_size}")
    logging.info(f"[execute_setup] Resource group: {resource_group}")  
    logging.info(f"[execute_setup] Storage resource group: {azure_storage_resource_group}") 
    logging.info(f"[execute_setup] Azure OpenAI resource group: {azure_aoai_resource_group}") 

    # local setting 
    search_index_name = os.getenv("SEARCH_INDEX_NAME")    
    search_index_interval = os.getenv("SEARCH_INDEX_INTERVAL")
    

    ###########################################################################
    # Get function key to be used later when creating the skillset
    ########################################################################### 
    function_key = get_function_key(subscription_id, resource_group, function_app_name, credential)
    if function_key is None:
            logging.error(f"Could not get function key. Please make sure the function {function_app_name}/document_chunking is deployed before running this script.")
            exit(1) 

    ###########################################################################
    # Approve Search Shared Private Links (if needed)
    ########################################################################### 
    logging.info("Approving search shared private links.")  
    approve_search_shared_private_access(subscription_id, resource_group, azure_storage_resource_group, azure_aoai_resource_group, function_app_name, storage_account_name, azure_openai_service_name, credential)

    ###########################################################################
    # Creating blob containers
    ###########################################################################
    # Note: this step was removed since the storage account and container are already created by azd provision

    ###############################################################################
    # Creating AI Search datasource
    ###############################################################################
    
    def create_datasource(search_service, search_api_version, datasource_name, storage_connection_string, container_name, credential, subfolder=None):
        body = {
            "description": f"Datastore for {datasource_name}",
            "type": "azureblob",
            "dataDeletionDetectionPolicy": {
                "@odata.type": "#Microsoft.Azure.Search.NativeBlobSoftDeleteDeletionDetectionPolicy"
            },
            "credentials": {
                "connectionString": storage_connection_string
            },
            "container": {
                "name": container_name,
                "query": f"{subfolder}/" if subfolder else ""  # Adding subfolder path if provided
            }
        }
        call_search_api(search_service, search_api_version, "datasources", f"{datasource_name}-datasource", "put", credential, body)

    logging.info("Creating datasources step.")
    start_time = time.time()

    # Define storage connection string without account key
    # TODO: Use storage account resource group
    storage_connection_string = f"ResourceId=/subscriptions/{subscription_id}/resourceGroups/{azure_storage_resource_group}/providers/Microsoft.Storage/storageAccounts/{storage_account_name}/;"

    # Creating main datasource
    create_datasource(search_service, search_api_version, f"{search_index_name}", storage_connection_string, storage_container, credential)

    # Creating NL2SQL datasource in different subfolders

    response_time = time.time() - start_time
    logging.info(f"Create datastores step. {round(response_time, 2)} seconds")


    ###############################################################################
    # Creating indexes
    ###############################################################################
    def create_index_body(index_name):
    # Load environment variables
        
        # Azure Cognitive Search settings
        service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
        api_version = "2024-11-01-preview" 
        admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        
        # Endpoint URL
        endpoint = f"https://{service_name}.search.windows.net/indexes/{index_name}?api-version={api_version}"
        
        # Headers
        headers = {
            'Content-Type': 'application/json',
            'api-key': admin_key
        }
        
        # Index definition
        body = {
            "name": index_name,
            "fields": [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": True,
                    "retrievable": True,
                    "stored": True,
                    "sortable": True,
                    "facetable": True,
                    "key": True,
                    "analyzer": "keyword",
                    "synonymMaps": []
                },
                {
                    "name": "parent_id",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "retrievable": True,
                    "stored": True,
                    "sortable": True,
                    "facetable": True,
                    "key": False,
                    "synonymMaps": []
                },
                {
                    "name": "metadata_storage_path",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": False,
                    "retrievable": True,
                    "stored": True,
                    "sortable": False,
                    "facetable": False,
                    "key": False,
                    "synonymMaps": []
                },
                {
                    "name": "metadata_storage_name",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": False,
                    "retrievable": True,
                    "stored": True,
                    "sortable": False,
                    "facetable": False,
                    "key": False,
                    "synonymMaps": []
                },
                {
                    "name": "chunk_id",
                    "type": "Edm.Int32",
                    "searchable": False,
                    "filterable": True,
                    "retrievable": True,
                    "stored": True,
                    "sortable": True,
                    "facetable": True,
                    "key": False,
                    "synonymMaps": []
                },
                {
                    "name": "content",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": True,
                    "retrievable": True,
                    "stored": True,
                    "sortable": True,
                    "facetable": True,
                    "key": False,
                    "analyzer": "standard",
                    "synonymMaps": []
                },
                {
                    "name": "page",
                    "type": "Edm.Int32",
                    "searchable": False,
                    "filterable": True,
                    "retrievable": True,
                    "stored": True,
                    "sortable": True,
                    "facetable": True,
                    "key": False,
                    "synonymMaps": []
                },
                {
                    "name": "offset",
                    "type": "Edm.Int64",
                    "searchable": False,
                    "filterable": False,
                    "retrievable": True,
                    "stored": True,
                    "sortable": True,
                    "facetable": True,
                    "key": False,
                    "synonymMaps": []
                },
                {
                    "name": "length",
                    "type": "Edm.Int32",
                    "searchable": False,
                    "filterable": False,
                    "retrievable": True,
                    "stored": True,
                    "sortable": True,
                    "facetable": True,
                    "key": False,
                    "synonymMaps": []
                },
                {
                    "name": "title",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": True,
                    "retrievable": True,
                    "stored": True,
                    "sortable": True,
                    "facetable": True,
                    "key": False,
                    "analyzer": "standard",
                    "synonymMaps": []
                },
                {
                    "name": "category",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": True,
                    "retrievable": True,
                    "stored": True,
                    "sortable": True,
                    "facetable": True,
                    "key": False,
                    "analyzer": "standard",
                    "synonymMaps": []
                },
                {
                    "name": "filepath",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": False,
                    "retrievable": True,
                    "stored": True,
                    "sortable": True,
                    "facetable": True,
                    "key": False,
                    "synonymMaps": []
                },
                {
                    "name": "url",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": False,
                    "retrievable": True,
                    "stored": True,
                    "sortable": True,
                    "facetable": True,
                    "key": False,
                    "synonymMaps": []
                },
                {
                    "name": "vector",
                    "type": "Collection(Edm.Single)",
                    "searchable": True,
                    "filterable": False,
                    "retrievable": True,
                    "stored": True,
                    "sortable": False,
                    "facetable": False,
                    "key": False,
                    "dimensions": 1536,
                    "vectorSearchProfile": "myHnswProfile",
                    "synonymMaps": []
                },
                {
                    "name": "keyPhrases",
                    "type": "Collection(Edm.String)",
                    "searchable": True,
                    "filterable": False,
                    "retrievable": True,
                    "stored": True,
                    "sortable": False,
                    "facetable": False,
                    "key": False,
                    "analyzer": "standard.lucene",
                    "synonymMaps": []
                },
                {
                    "name": "languageCode",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": True,
                    "retrievable": True,
                    "stored": True,
                    "sortable": False,
                    "facetable": False,
                    "key": False,
                    "synonymMaps": []
                },
                {
                    "name": "languageName",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": True,
                    "retrievable": True,
                    "stored": True,
                    "sortable": False,
                    "facetable": False,
                    "key": False,
                    "synonymMaps": []
                },
                {
                    "name": "organization_id",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": True,
                    "retrievable": True,
                    "stored": True,
                    "sortable": False,
                    "facetable": False,
                    "key": False,
                    "analyzer": "standard.lucene",
                    "synonymMaps": []
                }
            ],
            "scoringProfiles": [
                {
                    "name": f"{index_name}-scoring-profile",
                    "functionAggregation": "sum",
                    "text": {
                        "weights": {
                            "content": 45,
                            "keyPhrases": 45,
                            "title": 5
                        }
                    },
                    "functions": []
                }
            ],
            "corsOptions": {
                "allowedOrigins": ["*"],
                "maxAgeInSeconds": 60
            },
            "suggesters": [],
            "analyzers": [],
            "normalizers": [],
            "tokenizers": [],
            "tokenFilters": [],
            "charFilters": [],
            "similarity": {
                "@odata.type": "#Microsoft.Azure.Search.BM25Similarity"
            },
            "semantic": {
                "configurations": [
                    {
                        "name": "my-semantic-config",
                        "prioritizedFields": {
                            "prioritizedContentFields": [
                                {
                                    "fieldName": "content"
                                }
                            ],
                            "prioritizedKeywordsFields": [
                                {
                                    "fieldName": "category"
                                }
                            ]
                        }
                    }
                ]
            },
            "vectorSearch": {
                "algorithms": [
                    {
                        "name": "myHnswConfig",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "metric": "cosine",
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500
                        }
                    }
                ],
                "profiles": [
                    {
                        "name": "myHnswProfile",
                        "algorithm": "myHnswConfig",
                        "vectorizer": "vector-ce-vectorizer"
                    }
                ],
                "vectorizers": [
                    {
                        "name": "vector-ce-vectorizer",
                        "kind": "azureOpenAI",
                        "azureOpenAIParameters": {
                            "resourceUri": os.getenv("AZURE_OPENAI_ENDPOINT"),
                            "deploymentId": "text-embedding-3-small",
                            "apiKey": os.getenv("AZURE_OPENAI_API_KEY"),
                            "modelName": "text-embedding-3-small"
                        }
                    }
                ],
                "compressions": []
            }
        }
        
        # First, try to delete the existing index if it exists
        try:
            delete_response = requests.delete(endpoint, headers=headers)
            print(f"Delete existing index response: {delete_response.status_code}")
        except Exception as e:
            print(f"Error deleting existing index: {e}")
        
        # Create the new index
        try:
            response = requests.put(endpoint, headers=headers, json=body)
            
            if response.status_code in [200, 201]:
                print("Index created successfully!")
                print(response.json())
            else:
                print(f"Error creating index. Status code: {response.status_code}")
                print(f"Error message: {response.text}")
                
        except Exception as e:
            print(f"Error creating index: {e}")
        
        return body
    
    # Create the main index
    create_index_body(search_index_name)


    ###########################################################################
    # 04 Creating AI Search skillsets
    ###########################################################################
    logging.info("04 Creating skillsets step.")
    start_time = time.time()

    body = { 
        "name": f"{search_index_name}-skillset-chunking",
        "description":"SKillset to do document chunking",
        "skills":[ 
            { 
                "@odata.type":"#Microsoft.Skills.Custom.WebApiSkill",
                "name":"docint-processing",
                "description":"Process content with document intelligence markdown notation",
                "context":"/document",
                "httpMethod":"POST",
                "timeout":"PT3M50S",
                "batchSize":1,
                "inputs":[ 
                    {
                        "name":"documentUrl",
                        "source":"/document/metadata_storage_path"
                    },     
                    {
                        "name": "documentContent",
                        "source": "/document/content"
                    },              
                    { 
                        "name":"documentSasToken",
                        "source":"/document/metadata_storage_sas_token"
                    },
                    { 
                        "name":"documentContentType",
                        "source":"/document/metadata_content_type"
                    }
                ],
                "outputs":[ 
                    {
                        "name":"docintContent",
                        "targetName":"docintContent"
                    }
                ]
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.LanguageDetectionSkill",
                "name": "Lenguage Detection",
                "description": "Skill to get the language of the document",
                "context": "/document/docintContent/pages/*",
                "inputs": [
                    {
                        "name": "text",
                        "source": "/document/docintContent/pages/*",
                        "inputs": []
                    }
                ],
                "outputs": [
                    {
                        "name": "languageCode",
                        "targetName": "languageCode"
                    },
                    {
                        "name": "languageName",
                        "targetName": "languageName"
                    },
                    {
                        "name": "score",
                        "targetName": "score"
                    }
                ]
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.KeyPhraseExtractionSkill",
                "name": "Key Phrase Extraction",
                "description": "Skill to get the key phrases of the document",
                "context": "/document/docintContent/pages/*",
                "defaultLanguageCode": "en",
                "maxKeyPhraseCount": 10,
                "inputs": [
                    {
                        "name": "text",
                        "source": "/document/docintContent/pages/*",
                        "inputs": []
                    },
                    {
                        "name": "languageCode",
                        "source": "/document/docintContent/pages/*/languageCode",
                        "inputs": []
                    }
                ],
                "outputs": [
                    {
                        "name": "keyPhrases",
                        "targetName": "keyPhrases"
                    }
                ]
            }
        ],
        "indexProjections": {
            "selectors": [
                {
                    "targetIndexName":f"{search_index_name}",
                    "parentKeyFieldName": "parent_id",
                    "sourceContext": "/document/docintContent/pages/*",
                    "mappings": [
                        {
                            "name": "url",
                            "source": "/document/docintContent/url",
                            "inputs": []
                        },
                        {
                            "name": "organization_id",
                            "source": "/document/organization_id",
                            "inputs": []
                        },
                        {
                            "name": "filepath",
                            "source": "/document/docintContent/url",
                            "inputs": []
                        },
                        {
                            "name": "content",
                            "source": "/document/docintContent/pages/*",
                            "inputs": []
                        },
                        {
                            "name": "vector",
                            "source": "/document/docintContent/pages/*/vector",
                            "inputs": []
                        },
                        {
                            "name": "metadata_storage_path",
                            "source": "/document/metadata_storage_path",
                            "inputs": []
                        },
                        {
                            "name": "metadata_storage_name",
                            "source": "/document/metadata_storage_name",
                            "inputs": []
                        },
                        {
                            "name": "keyPhrases",
                            "source": "/document/docintContent/pages/*/keyPhrases",
                            "inputs": []
                        },
                        {
                            "name": "languageCode",
                            "source": "/document/docintContent/pages/*/languageCode",
                            "inputs": []
                        },
                        {
                            "name": "languageName",
                            "source": "/document/docintContent/pages/*/languageName",
                            "inputs": []
                        }                      
                    ]
                }
            ],
            "parameters": {
                "projectionMode": "skipIndexingParentDocuments"
            }
        }
    }
    if azure_search_use_mis:
        body['skills'][0]['uri'] = f"{function_endpoint}/api/document-chunking"
        body['skills'][0]['authResourceId'] = f"api://{search_principal_id}"
    else:
        body['skills'][0]['uri'] = f"{function_endpoint}/api/document-chunking?code={function_key}"
        

    # first delete to enforce web api skillset to be updated
    call_search_api(search_service, search_api_version, "skillsets", f"{search_index_name}-skillset-chunking", "delete", credential)        
    call_search_api(search_service, search_api_version, "skillsets", f"{search_index_name}-skillset-chunking", "put", credential, body)

    response_time = time.time() - start_time
    logging.info(f"04 Create skillset step. {round(response_time,2)} seconds")

    ###########################################################################
    # 05 Creating indexers
    ###########################################################################
    # create the indexers for the main index
    def create_indexer_body(indexer_name):
            # Azure Cognitive Search settings from .env
        service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
        api_version = "2024-11-01-preview" 
        admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        indexer_name = f"{search_index_name}-indexer-chunk-documents"
        # Endpoint URL
        endpoint = f"https://{service_name}.search.windows.net/indexers/{indexer_name}?api-version={api_version}"
        
        # Headers
        headers = {
            'Content-Type': 'application/json',
            'api-key': admin_key
        }
        body = {
        "name": indexer_name,
        "dataSourceName": f"{search_index_name}-datasource",
        "targetIndexName": f"{search_index_name}",
        "skillsetName": f"{search_index_name}-skillset-chunking",
        "schedule": {"interval": f"{search_index_interval}"},
        "fieldMappings": [
            {
                "sourceFieldName": "metadata_storage_path",
                "targetFieldName": "id",
                "mappingFunction": {
                    "name": "base64Encode"
                }
            },
            {
                "sourceFieldName": "organization_id",
                "targetFieldName": "organization_id",
                "mappingFunction": None
            }
        ],
        "outputFieldMappings": [],
        "parameters": {
            "batchSize": 1,
            "maxFailedItems": -1,
            "maxFailedItemsPerBatch": -1,
            "configuration": {
                "dataToExtract": "contentAndMetadata",
                "parsingMode": "default"
            }
        }
    }
        # First, try to delete the existing indexer if it exists
        try:
            delete_response = requests.delete(endpoint, headers=headers)
            print(f"Delete existing indexer response: {delete_response.status_code}")
        except Exception as e:
            print(f"Error deleting existing indexer: {e}")
        
        # Create the new indexer
        try:
            response = requests.put(endpoint, headers=headers, json=body)
            
            if response.status_code in [200, 201]:
                print("Indexer created successfully!")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"Error creating indexer. Status code: {response.status_code}")
                print(f"Error message: {response.text}")
                
        except Exception as e:
            print(f"Error creating indexer: {e}")

    

def main(subscription_id=None, resource_group=None, function_app_name=None, search_principal_id='', azure_search_use_mis=False, enable_managed_identities=False, enable_env_credentials=False):
    """
    Sets up a chunking function app in Azure.

    Args:
        subscription_id (str): The subscription ID to use. If not provided, the user will be prompted to enter it.
        resource_group (str): The resource group to use. If not provided, the user will be prompted to enter it.
        function_app_name (str): The name of the chunking function app. If not provided, the user will be prompted to enter it.
        search_principal_id (str): Entra ID of the search managed identity.         
        azure_search_use_mis (bool): Whether to use Search Service Managed Identity to Connect to data ingestion function
        enable_managed_identities (bool, optional): Whether to use VM's managed identities to run the setup, defaults to False.
        enable_env_credentials (bool, optional): Whether to use environment credentials to run the setup, defaults to False.
    """   
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Starting setup.")
    if subscription_id is None:
        subscription_id = input("Enter subscription ID: ")
    if resource_group is None:
        resource_group = input("Enter function app resource group: ")
    if function_app_name is None:
        function_app_name = input("Enter chunking function app name: ")

    start_time = time.time()

    execute_setup(subscription_id, resource_group, function_app_name, search_principal_id, azure_search_use_mis, enable_managed_identities, enable_env_credentials)

    response_time = time.time() - start_time
    logging.info(f"Finished setup. {round(response_time,2)} seconds")

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')    
    # parser = argparse.ArgumentParser(description='Script to do the data ingestion setup for Azure Cognitive Search.')
    # parser.add_argument('-s', '--subscription_id', help='Subscription ID')
    # parser.add_argument('-r', '--resource_group', help='Resource group (Function App)')
    # parser.add_argument('-f', '--function_app_name', help='Chunking function app name')
    # parser.add_argument('-a', '--search_principal_id', default='none', help='Entra ID of the search service')
    # parser.add_argument('-m', '--azure_search_use_mis', help='Use Search Service Managed Identity to Connect to data ingestion function')
    # parser.add_argument('-i', '--enable_managed_identities', action='store_true', default=False, help='Use VM\'s managed identities for the setup')
    # parser.add_argument('-e', '--enable_env_credentials', action='store_true', default=False, help='Use environment credentials for the setup')    
    # args = parser.parse_args()

    # # format search_use_mis to boolean
    # search_use_mis = args.azure_search_use_mis.lower() == "true" if args.azure_search_use_mis not in [None, ""] else False

    # # Log all arguments
    # logging.info(f"[main] Subscription ID: {args.subscription_id}")
    # logging.info(f"[main] Resource group: {args.resource_group}") 
    # logging.info(f"[main] Function app name: {args.function_app_name}")
    # logging.info(f"[main] Search principal ID: {args.search_principal_id}")
    # logging.info(f"[main] Azure Search use MIS: {search_use_mis}")
    # logging.info(f"[main] Enable managed identities: {args.enable_managed_identities}")
    # logging.info(f"[main] Enable environment credentials: {args.enable_env_credentials}")

    # main(subscription_id=args.subscription_id, resource_group=args.resource_group, function_app_name=args.function_app_name, search_principal_id=args.search_principal_id, 
    #     azure_search_use_mis=search_use_mis, enable_managed_identities=args.enable_managed_identities, enable_env_credentials=args.enable_env_credentials)

    create_indexer_body(indexer_name="nam-indexer", index_name="ragindex")

