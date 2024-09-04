from config import settings
from elasticsearch import Elasticsearch

elastic_client = None

vietnamese_index_settings = {
    "settings": {
        "analysis": {
            "analyzer": {
                "my_vi_analyzer": {
                    "tokenizer": "vi_tokenizer",
                    "filter": [
                        "lowercase",
                        "ascii_folding",
                        "my_word_delimiter_graph",
                        "flatten_graph",  # required
                    ],
                }
            },
            "filter": {
                "ascii_folding": {"type": "asciifolding", "preserve_original": True},
                "my_word_delimiter_graph": {
                    "type": "word_delimiter_graph",
                    "preserve_original": True,
                },
            },
        }
    }
}


if settings.elastic_password is not None:
    elastic_client = Elasticsearch(
        hosts=settings.elastic_endpoint,
        basic_auth=(settings.elastic_username, settings.elastic_password),
    )

    # if not elastic_client.indices.exists(index="videos"):
    #     elastic_client.indices.create(index="videos", body=index_settings)
    #     new_init = True
else:
    print("elasticsearch password not provided")