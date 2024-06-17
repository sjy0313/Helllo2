package ch03_.operator.sec03_over_underflow;
// overflow : 타입이 허용하는 한 최대값을 벗어나는 것 
//underflow : 타입이 허용하는 한 최소값을 벗어나는 것
public class OverflowUnderflowExample {
	public static void main(String[] args) {
		
		System.out.printf("바이트: min : %d\n", Byte.MIN_VALUE);
		System.out.printf("바이트: max : %d\n", Byte.MAX_VALUE);
		
		byte var1 = 125; // -128~127
		
		System.out.println("[overflow]");
		
		for(int i=0; i<5; i++) { //{ }를 5번 반복 실행
			var1++; //++ 연산은 var1의 값을 1 증가시킨다.
			System.out.println("var1: " + var1);
		}

		
		System.out.println("-----------------------");
		System.out.println("[underflow]");
		byte var2 = -125;
		for(int i=0; i<5; i++) { //{ }를 5번 반복 실행
			var2--; //-- 연산은 var2의 값을 1 감소시킨다.
			System.out.println("var2: " + var2);
		}
	}
}