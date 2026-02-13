from langgraph.checkpoint.memory import InMemorySaver
from typing import List, Optional, Literal, Dict, Any, Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langgraph.graph.message import add_messages


class LibraryState(TypedDict):
    # TODO: Add a messages field that stores a List[AnyMessage] with an add_messages reducer
    messages: Annotated[List[AnyMessage], add_messages]
    section: Optional[Literal['borrow', 'return', 'overdue', 'unknown']]
    # TODO: Add a books_borrowed field that stores a List[str] without a reducer
    books_borrowed: List[str]
    resolved: bool
    #TODO: Add a last_user_message field to store the last human message received
    last_user_message: str

# Router: return a partial update dict
def route_library(state: LibraryState) -> Dict[str, Any]:
    # Find the last human message
    last_msg = ''
    for msg in reversed(state.get('messages', [])):
        if isinstance(msg, HumanMessage):
            last_msg = msg.content.lower()
            break

    if 'borrow' in last_msg:
        intent = 'borrow'
    elif 'return' in last_msg:
        intent = 'return'
    elif 'overdue' in last_msg or 'fine' in last_msg:
        intent = 'overdue'
    else:
        intent = 'unknown'

    # Return state updates
    return {
        'last_user_message': last_msg,
        'section': intent,
        'resolved': False
    }

# Common Function
def find_title(message: str, category: str) -> str | None:
    words = message.split()
    for i, word in enumerate(words):
        if word == category and i + 1 < len(words):
            return words[i + 1]
    return None
# Borrow handler
def handle_borrow(state: LibraryState) -> Dict[str, Any]:
    # TODO: extract the title after 'borrow'
    # Build and return a new dict with updates:
    # - 'books_borrowed': a new list with the new title appended if not already borrowed
    # - 'messages': a list containing one AIMessage response
    # - 'resolved': True

    book_title = None
    for msg in reversed(state.get('messages', [])):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if 'borrow' in content.lower():
                # Extract book title after 'borrow'
                book_title = content.split('borrow', 1)[1].strip().title()
                break

    current_books = state.get('books_borrowed', [])
    updates = {}

    if book_title:
        if book_title not in current_books:
            # Replace the entire list with updated version
            updates['books_borrowed'] = current_books + [book_title]
            ai_text = f"Sure! I've added '{book_title}' to your borrowed books."
        else:
            ai_text = f"It looks like '{book_title}' is already on your list."
    else:
        ai_text = "I'm sorry, I couldn't identify the book you'd like to borrow."

    return {
        'messages': [AIMessage(content=ai_text)],
        **updates,
        'resolved': True
    }


# Return handler
def handle_return(state: LibraryState) -> Dict[str, Any]:
    # TODO: extract the title from the user's message after the word: 'return'
    # Build and return a dict:
    # - If the title is in books_borrowed, provide a new 'books_borrowed' list without it
    # - Always include 'messages' with an AIMessage response and set 'resolved': True
    book_title = None
    for msg in reversed(state.get('messages', [])):  # Fixed: use .get() with default
        if isinstance(msg, HumanMessage):
            content = msg.content
            if 'return' in content.lower():
                # Extract book title after 'return'
                book_title = content.split('return', 1)[1].strip().title()
                break

    current_books = state.get('books_borrowed', [])

    if book_title and book_title in current_books:
        # Create a new list without the returned book (no direct mutation)
        updated_books = [book for book in current_books if book != book_title]
        ai_text = f"Thank you! I've removed '{book_title}' from your borrowed books."

        # Return the complete new list to replace the old one
        return {
            'messages': [AIMessage(content=ai_text)],
            'books_borrowed': updated_books,  # Replace entire list
            'resolved': True
        }
    elif book_title:
        ai_text = f"It doesn't look like you borrowed '{book_title}' from us."
    else:
        ai_text = "I'm sorry, I couldn't identify the book you're returning."

    return {
        'messages': [AIMessage(content=ai_text)],
        'resolved': True
    }


# Overdue handler
def handle_overdue(state: LibraryState) -> Dict[str, Any]:
    # TODO: return a dict with an AIMessage about the number of borrowed books and set 'resolved': True
    current_books = state.get('books_borrowed', [])

    if current_books:
        books_list = ', '.join([f"'{book}'" for book in current_books])
        ai_text = f"You currently have {len(current_books)} book(s) borrowed: {books_list}. Please visit the library to check if any fines apply."
    else:
        ai_text = "You have no books borrowed at the moment. There are no overdue fines."

    return {
        'messages': [AIMessage(content=ai_text)],
        'resolved': True
    }


# Unknown handler
def handle_unknown(state: LibraryState) -> Dict[str, Any]:
    # TODO: return a dict with an AIMessage asking for clarification and set 'resolved': True
    return {
        'messages': [AIMessage(content=f"I'm sorry. I didn't quite catch that. Can you clarify?")],
        'resolved': True
    }

# Decide next step
def next_step(state: LibraryState) -> str:
    if state.get('resolved', False):
        return END
    section = state.get('section', None)
    return section if section else END

# Build workflow
workflow = StateGraph(LibraryState)
workflow.add_node('router', route_library)
workflow.add_node('borrow', handle_borrow)
workflow.add_node('return', handle_return)
workflow.add_node('overdue', handle_overdue)
workflow.add_node('unknown', handle_unknown)

# TODO Add edge to connect START to 'router'
workflow.add_edge(START, 'router')
# Conditional routing
workflow.add_conditional_edges('router', next_step, {
    'borrow': 'borrow',
    'return': 'return',
    'overdue': 'overdue',
    'unknown': 'unknown',
    END: END
})

#TODO: Connect each handler to the END node
workflow.add_edge('borrow', END)
workflow.add_edge('return', END)
workflow.add_edge('overdue', END)
workflow.add_edge('unknown', END)




if __name__ == "__main__":
    # TODO: compile with InMemorySaver and test multiple invocations with same thread_id
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": 'demo_user'}}

    # First interaction: borrow a book
    state1 = {
        'messages': [HumanMessage(content="I want to borrow Moby Dick")],
        'books_borrowed': [],
        'resolved': False
    }
    result1 = app.invoke(state1, config=config)
    print("Result 1 - Borrow:")
    print(f"  User message: {result1.get('last_user_message', '')}")
    print(f"  Books borrowed: {result1.get('books_borrowed', [])}")
    print(f"  Last message: {result1['messages'][-1].content}\n")

    # Second interaction: check overdue (properly create new state)
    current_state = app.get_state(config).values
    state2 = {
        **current_state,
        'messages': current_state['messages'] + [HumanMessage(content="Are there any overdue books?")]
    }
    result2 = app.invoke(state2, config=config)
    print("Result 2 - Check overdue:")
    print(f"  User message: {result2.get('last_user_message', '')}")
    print(f"  Books borrowed: {result2.get('books_borrowed', [])}")
    print(f"  Last message: {result2['messages'][-1].content}\n")

    # Third interaction: return the book (properly create new state)
    current_state = app.get_state(config).values
    state3 = {
        **current_state,
        'messages': current_state['messages'] + [HumanMessage(content="I need to return Moby Dick")]
    }
    result3 = app.invoke(state3, config=config)
    print("Result 3 - Return:")
    print(f"  User message: {result3.get('last_user_message', '')}")
    print(f"  Books borrowed: {result3.get('books_borrowed', [])}")
    print(f"  Last message: {result3['messages'][-1].content}\n")

    # Fourth interaction: check status again
    current_state = app.get_state(config).values
    state4 = {
        **current_state,
        'messages': current_state['messages'] + [HumanMessage(content="Do I have any overdue books?")]
    }
    result4 = app.invoke(state4, config=config)
    print("Result 4 - Final check:")
    print(f"  User message: {result4.get('last_user_message', '')}")
    print(f"  Books borrowed: {result4.get('books_borrowed', [])}")
    print(f"  Last message: {result4['messages'][-1].content}")
