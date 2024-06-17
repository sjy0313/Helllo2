package ch04_.statement.sec05.whileloop;
// class이름과 파일이름은 같아야함.
// escaping loop
public class SumFrom1To100Example3 {
	
	public static void main(String[] args) {
		int sum = 0;
		int i = 1;
		
		while(true) { // 무한루프
			if(i > 100) {
				break;
			}
			sum += i;
			i++;
		}
		// Unreachable code
		System.out.println("1~" + (i-1) + " 합 : " + sum);
	}
	
}