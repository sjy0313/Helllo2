package ch04_.statement.sec05.whileloop;
// class이름과 파일이름은 같아야함.
public class SumFrom1To100Example2 {
	
	public static void main(String[] args) {
		int sum = 0;
		int i = 1;
		
		while(true) { // 무한루프
			sum += i;
			i++;
		}
		// Unreachable code
		System.out.println("1~" + (i-1) + " 합 : " + sum);
	}
}