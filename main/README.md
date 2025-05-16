**In terminal run:**
1. cd to the `\SkinToneAdvisor` folder
2. run 
```shell
colorenv\Scripts\activate
```
to activate virtual env
3. Split into 2 terminals, do the same for another terminal to activate the virtual env
```shell
colorenv\Scripts\activate
```
4. cd to `\main` directory in both terminals
4. In 1 terminal run:
```shell
python -m uvicorn main:app --reload
```
5. In another, run 
```shell
streamlit run .\app.py
```