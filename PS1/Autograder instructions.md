Hello everyone!

Please submit the report and code to Gradescope.

While we won't give away too much information about the exact code running in the autograder, we will include a brief description of what each test is analyzing. The main goal here is to try to steer you in the right direction so that you have an easier time getting the tests to pass.

#### extract_green and extract_red:

- A 3-channel (rows, cols, 3) array with random numbers is created. 
- Your extract_green / extract_red method is called and the result is compared against the correct color plane. 
- The test passes if all the elements in your output array match.

#### swap_green_blue:

- A 3-channel (rows, cols, 3) array with random numbers is created. 
- Your swap_green_blue function is called and the output is collected.
- The output is compared to the correct color planes from the original array.

#### copy_paste_middle:

- Two 1-channel arrays are generated with random numbers.
- The output from copy_paste_middle is verified against a reference solution.
- The test passes if all elements match.

#### copy_paste_middle_circle:

- Two 1-channel arrays are generated with random numbers.
- The output from copy_paste_middle_circle is verified against a reference solution.
- The test passes if all elements match.
- HINT: Remember that the center pixel counts as 1. If the required radius 1, diameter is 3 and not 2. Think pixels. A dot is 1 pixel.

#### image_stats:

- Here there are 4 tests that come from this function: min, max, mean, standard deviation.
- For each test, one 1-channel arrays are generated with random numbers is generated.
- Your output is checked using a reference solution.
- Each test will pass if it accurately matches the correct value.

#### center_and_normalize:

- A 1-channel array and a scale factor are generated using random numbers.
- Given that the output of your normalized method modifies the standard deviation of the original array, the test checks if the standard deviation of the output array is close to the scale factor.

#### shift_image_left:

- A 1-channel array and a shift factor are generated using random numbers.
- Your method shift_image_left is called and the output is verified against a reference solution.
- The test will pass if all the array values match the sample answer.

#### difference_image:

- A 1-channel array and a shift factor are generated using random numbers.
- A shifted version of the array above is created and passed to your method differenceImage.
- Your output is evaluated against a reference solution.
- The test will pass if all the array values match the sample answer.

#### add_noise:

- A 3-channel array and a standard deviation constant (sigma) are generated using random numbers. 
- A color plane is picked at random.
- These three items are passed to your add_noise function.
- Your output is verified in such way that it should present a standard deviation value (in the selected channel) close to the input parameter.

We want to reiterate our advice about using Numpy indexing methods that can speed up your code. Nested for loops tend to be very slow in Python so try to avoid them whenever you can. 

If you receive a timeout message it means you need to optimize your code a little. Additionally, Gradescope has a global quota that is max 10 submissions in a 60 minute interval.

Remember that you should not post code or parameters on Piazza, instead send us a private post with it so we can guide you better. Also, make sure you search this forum to see whether your question has been answered before posting.

Please ask Autograder related technical issues in the megathread id edstem
