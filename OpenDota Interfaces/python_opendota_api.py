import time
import python_opendota
from pprint import pprint
from python_opendota.apis.tags import benchmarks_api
# Defining the host is optional and defaults to http://api.opendota.com/api
# See configuration.py for a list of all supported configuration parameters.
configuration = python_opendota.Configuration(
    host = "http://api.opendota.com/api"
)


# Enter a context with an instance of the API client
with python_opendota.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = benchmarks_api.BenchmarksApi(api_client)
    hero_id = "hero_id_example" # str | Hero ID

    try:
        # GET /benchmarks
        api_response = api_instance.benchmarks_get(hero_id)
        pprint(api_response)
    except python_opendota.ApiException as e:
        print("Exception when calling BenchmarksApi->benchmarks_get: %s\n" % e)