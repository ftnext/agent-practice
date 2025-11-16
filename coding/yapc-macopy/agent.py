# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai",
#     "rich",
# ]
# ///
import argparse
import io
import os
import re
from contextlib import redirect_stdout

from openai import OpenAI
from rich import print as rich_print

client = OpenAI(
    api_key=os.environ["SAKURA_AI_API_KEY"],
    base_url="https://api.ai.sakura.ad.jp/v1",
)


def chat_completion(messages):
    response = client.chat.completions.create(
        model="Qwen3-Coder-480B-A35B-Instruct-FP8",
        messages=messages,
        max_tokens=100_000,
        temperature=0.0,
        tool_choice="none",
        tools=[],
        stream=False,
    )
    return response.choices[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    args = parser.parse_args()

    messages = [
        {
            "role": "system",
            "content": "あなたはなんでもPythonで解決してしまう親切なAIアシスタントです",
        },
        {
            "role": "user",
            "content": f"""\
Your task: {args.task}. 3つのbacktickで囲まれたコードブロック内にPythonのコードを使って返信してください。
コードブロック内をexecで実行した際の標準出力があなたに渡されます。
あなたはコードを繰り返し実行して、その結果を読み取り、さらに必要があれば結果を元にコードを生成します。タスクが完了するまで繰り返します。
もしタスクが完了した時、コードブロックを実行した際の**標準出力の最初の行**に'__COMPLETE__'を出力してください。ユーザーにはその後の行が最終結果として表示されます。

Pythonコードブロックの前にはTHOUGHTセクションを追加し、推論プロセスを説明してください。
以下の<format_example>セクションに示すようなフォーマットにしてください。

<format_example>
THOUGHT: Your reasoning and analysis here

```python
# Your Python code here
```

この形式の応答に従わない場合は、あなたの応答は拒否されます。
""",
        },
    ]
    while True:
        choice = chat_completion(messages)
        completion = choice.message.content
        print(completion)  # See THOUGHT
        messages.append({"role": "assistant", "content": completion})
        if match := re.search(r"```python\s*(.*?)\s*```", completion, re.DOTALL):
            code = match.group(1)
            rich_print(f"[green]実行:\n{code}[/green]")
            f = io.StringIO()
            try:
                with redirect_stdout(f):
                    result = exec(code)
                    if result is None:
                        result = f.getvalue()
            except Exception as ex:
                rich_print(f"[red]コードの実行中にエラーが発生しました: {ex!r}[/red]")
                messages.append(
                    {
                        "role": "user",
                        "content": f"コードの実行中にエラーが発生しました: {ex!r}",
                    }
                )
            else:
                if result.startswith("__COMPLETE__"):
                    print(result.removeprefix("__COMPLETE__").lstrip())
                    break
                rich_print(f"[blue]{result}[/blue]")
                messages.append({"role": "user", "content": result})
        else:
            rich_print(f"[red]コードブロックが見つかりませんでした: {completion}[/red]")
            break
