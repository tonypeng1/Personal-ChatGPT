# import streamlit as st

# st.title('Counter Example')
# if 'count' not in st.session_state:
#     st.session_state.count = 0

# increment = st.button('Increment')
# if increment:
#     st.session_state.count += 1

# st.write('Count = ', st.session_state.count)



# import streamlit as st

# st.title('Counter example using callbacks')
# if 'count' not in st.session_state:
#     st.session_state.count = 0

# def increment_counter():
#     st.session_state.count += 1

# st.button('Increment', on_click=increment_counter)

# st.write('Count = ', st.session_state.count)



# import streamlit as st

# st.title('Counter example using callbacks with args')
# if 'count' not in st.session_state:
#     st.session_state.count = 0

# increment_value = st.number_input('Enter an increment value', value=0, step=1)

# def increment_counter(increment_value):
#     st.session_state.count += increment_value

# st.button('Increment', on_click=increment_counter, args=(increment_value,)) # Note the args has to be
# # an iterable, not a integer)

# st.write('Counter = ', st.session_state.count)



# import streamlit as st

# st.title('Counter example using callbacks with args')
# if 'count' not in st.session_state:
#     st.session_state.count = 0

# def increment_counter(increment_value=0):
#     st.session_state.count += increment_value

# def decrement_counter(decrement_value=0):
#     st.session_state.count -= decrement_value

# increment_value_set = 5
# decrement_value_set = 1

# st.write('Increment value =', increment_value_set)
# st.button('Increment', on_click=increment_counter, kwargs=dict(increment_value=increment_value_set))

# st.write('Decrement value = ', decrement_value_set)
# st.button('Decrement', on_click=decrement_counter, kwargs=dict(decrement_value=decrement_value_set))

# st.write('Counter = ', st.session_state.count)



import streamlit as st

st.title('Counter example using callbacks with args')
if 'count' not in st.session_state:
    st.session_state.count = 0

def increment_counter(increment_value=0):
    st.session_state.count += increment_value

def decrement_counter(decrement_value=0):
    st.session_state.count -= decrement_value

increment_value_set = 5
decrement_value_set = 1

st.write('Increment value =', increment_value_set)
st.button('Increment', on_click=increment_counter, kwargs=dict(increment_value=increment_value_set))

st.write('Decrement value = ', decrement_value_set)
st.button('Decrement', on_click=decrement_counter, kwargs=dict(decrement_value=decrement_value_set))

st.write('Counter = ', st.session_state.count)