import pdb
### NAV
_NAV_SYSTEM = """You are an assistant trained to navigate the {_APP} screen. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
{_ACTION_SPACE}
"""
_NAV_FORMAT = """
Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

app_map = {
'web-single': 'website',
'web-multi': 'website',
'smartphone': 'smartphone'
}
action_map = {
'web-single': """
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
3. `SELECT`: Select a value for an element, value is not applicable and the position [x,y] is required. 
4. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
5. `ANSWER`: Answer the question, value is the answer and the position is not applicable.
6. `ENTER`: Enter operation, value and position are not applicable.
7. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
8. `SELECT_TEXT`: Select some text content, value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the select operation.
9. `COPY`: Copy the text, value is the text to copy and the position is not applicable.
""",

'web-multi': """
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
3. `SELECT`: Select a value for an element, value is not applicable and the position [x,y] is required. 
4. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
5. `ANSWER`: Answer the question, value is the answer and the position is not applicable.
6. `ENTER`: Enter operation, value and position are not applicable.
7. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
8. `SELECT_TEXT`: Select some text content, value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the select operation.
9. `COPY`: Copy the text, value is the text to copy and the position is not applicable.
""",

'smartphone': """
1. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is not applicable.
2. `SWIPE`: Swipe the screen, value is not applicable and the position [[x1,y1], [x2,y2]] is the start and end position of the swipe operation.
3. `TAP`: Tap on an element, value is not applicable and the position [x,y] is required.
4. `ANSWER`: Answer the question, value is the status (e.g., 'task complete') and the position is not applicable.
5. `ENTER`: Enter operation, value and position are not applicable.
"""
}

_NAV_USER = """{system}
Task: {task}
Observation: <|image_1|>
Action History: {action_history}
What is the next action?
"""

def navigation_to_qwen(task, image, action_history, split, answer_dict=None, skip_readme=False):
    transformed_data = []
    user_content = []
    if not skip_readme:
        system_prompt = _NAV_SYSTEM.format(_APP=app_map[split], _ACTION_SPACE=action_map[split])
    else:
        system_prompt = ""
    system_prompt += _NAV_FORMAT

    user_content.append({"type": "text", "text": system_prompt})
    user_content.append({"type": "text", "text": f"Task: {task}\n"})
    user_content.extend(action_history)

    transformed_data.append(
                {
                    "role": "user",
                    "content": user_content,
                },
            )
    return transformed_data
