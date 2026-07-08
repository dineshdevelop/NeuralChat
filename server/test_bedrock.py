import boto3
import json
from dotenv import load_dotenv
load_dotenv()

def test():
    try:
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
        body = json.dumps({"prompt": "hi", "max_gen_len": 50, "temperature": 0.5, "top_p": 0.9})
        response = client.invoke_model(
            modelId='meta.llama3-8b-instruct-v1:0',
            body=body
        )
        print("Success:", response['ResponseMetadata']['HTTPStatusCode'])
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
