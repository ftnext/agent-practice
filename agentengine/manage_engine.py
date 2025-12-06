# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "google-cloud-aiplatform[adk,agent-engines]>=1.111",
# ]
# ///
import asyncio

import vertexai

PROJECT_ID = "adk-practice-480404"
LOCATION = "asia-northeast1"

client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

for agent in client.agent_engines.list():
    print(agent.api_resource.name)
    break

adk_app = client.agent_engines.get(name=agent.api_resource.name)


async def main(message):
    async for event in adk_app.async_stream_query(user_id="user", message=message):
        print(event)


if __name__ == "__main__":
    asyncio.run(main("最近のAIニュースにはどんなものがある？"))
