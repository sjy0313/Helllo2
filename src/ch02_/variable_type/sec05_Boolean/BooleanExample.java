package ch02_.variable_type.sec05_Boolean;

public class BooleanExample {
	public static void main(String[] args) { 
		// 논리타입 : boolean : true , false 
		boolean stop = true;
		if(stop) {
			System.out.println("중지합니다.");
		} else {
			System.out.println("시작합니다.");
		}

		int x = 10;
		boolean result1 = (x == 20); //변수 x의 값이 20인가?
		boolean result2 = (x != 20); //변수 x의 값이 20이 아닌가?
		System.out.println("result1: " + result1); // result1: false
		System.out.println("result2: " + result2); // result2: true
	 }
 }