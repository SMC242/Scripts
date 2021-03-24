from typing import Any, List, Tuple


# Edit these constants to change the output query

# Put any text in this constant
LYRICS = """I'm a little teapot
Short and stout
Here is my handle
Here is my spout
When I get all steamed up
Hear me shout
Tip me over and pour me out!

I'm a very special teapot
Yes, it's true
Here's an example of what I can do
I can turn my handle into a spout
Tip me over and pour me out!"""

# Put any query that ends with a c: command and a limit WITHOUT the command
# See below for an example
QUERY = "https://census.daybreakgames.com/get/ps2:v2/character?name.first=benmitchellmtbV5&c:join=profile&limit=50"


def pipe(*funcs):
    def pipe_inner(*initial_args, **initial_kwargs):
        result = funcs[0](*initial_args, **initial_kwargs)
        for func in funcs[1:]:
            result = func(result)
        return result
    return pipe_inner


def remove_apostraphes(text: str) -> str:
    return text.replace("'", "")


def remove_commas(text: str) -> str:
    return text.replace(",", "")


def remove_bangs(text: str) -> str:
    return text.replace("!", "")


def remove_dots(text: str) -> str:
    return text.replace(".", "")


def remove_request_keywords(text: str) -> str:
    return text.replace("get", "").replace("post", "")


def remove_newlines(text: str) -> str:
    return text.replace("\n", " ")


clean_text = pipe(
    remove_apostraphes,
    remove_commas,
    remove_bangs,
    remove_request_keywords,
    remove_newlines,
)


def to_words(text: str) -> List[str]:
    return text.split(" ")


clean_words = pipe(clean_text, to_words)


def half_list(xs: List[Any]) -> Tuple[List[Any], List[Any]]:
    half = len(xs) // 2
    return (xs[:half], xs[half:])


def with_slashes(words: List[str]) -> List[str]:
    return [f"{w}/" for w in words]


def slash_list(slashed_words: List[str]) -> str:
    return "".join(slashed_words)


slashed_text = pipe(clean_words, with_slashes, slash_list)


def url_template(api_query: str):
    """The query must end with a c: command and limit={any_number}
    E.G c:join=character&limit=50"""
    def query_to_insertables(query: str) -> List[str]:
        """You can insert stuff between the domain and /get/
        and at the end of the query when it has a limit"""
        head, tail = query.split("get")
        if isinstance(tail, list):
            raise ValueError("The query should not have >1 instance of /get/")
        return [head, "/get", tail]

    def url_template_inner(to_insert: Tuple[str, str]):
        portions = query_to_insertables(api_query)
        return portions[0] + to_insert[0] + portions[1] + portions[2] + to_insert[1]
    return url_template_inner


insert_slashed_lyrics = pipe(
    slashed_text,
    half_list,
    url_template(QUERY),
)

print(insert_slashed_lyrics(LYRICS))
