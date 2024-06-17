package ch04_.statement.sec05.whileloop;
// class이름과 파일이름은 같아야함.
public class SumFrom1To100Example {
	
	public static void main(String[] args) {
		int sum = 0;
		
		
		int i = 1;
		
		while(i<=100) {
			sum += i;
			i++;
		}

		System.out.println("1~" + (i-1) + " 합 : " + sum);
	}
}