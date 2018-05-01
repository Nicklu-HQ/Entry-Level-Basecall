    
We are glad to provide an entry-level basecaller software which name is ‘e2e’ (from end to end, from current signal to base sequence).

‘e2e’ is a simplified version of ‘Chiron’ (https://github.com/haotianteng/Chiron).
We also simplified the data input (from Events which is produced by albacore). Although the labeled base aren’t 100% accuracy, it’s still work. Also you can align the labeled base sequence to the reference sequence to get the true labeled base. 

Trained within 30 mins on desktop computer with a small training set from Lambda DNA. The accuracy is close to 80%. It means the regular parts of current signal are easy to recognized. 

‘e2e’ is written by Python and based on tensorflow. You can replace your own data or change the structure.

What can we do with a customized basecaller?

1.	Improve the accuracy with you own data

2.	Help to detect the modification of DNA or RNA 
    Since we can learn the base information from current signal with a ‘classifier’, we can also simulate current signal from base with a ‘generator’. Matching the real signal with the artificial signal lead a modification detectation. 

3.	Improve the consensus 
    A traditional consensus algorithm takes the information of the sequence ‘AGCT’. A customized basecaller will give more information (for example the interval and the possibility of ‘AGCT’) 
