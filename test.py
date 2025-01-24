import csv

def student_list():
    # Open the CSV file in read mode
    with open('Student_list/Student-list.csv', mode='r') as file:
        data = []
        reader = csv.reader(file)
        
        i = 0
        # Loop through the rows and collect data
        for row in reader:
            if i != 0:  # Skip header
                data.append(row)
            i += 1
        return data

def find_student(student_id):
    # Get the student list
    students = student_list()
    
    # Iterate through the student list to find the student by ID
    for student in students:
        if student_id in student:  # Assuming student_id is part of the row
            return student
    return None  # Return None if not found

# Search for the student with ID 'e20200008'
student = find_student('e20200008')

if student:
    print(f"Student found: {student}")
else:
    print("Student not found.")
