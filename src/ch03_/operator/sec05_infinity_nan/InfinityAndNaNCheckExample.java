package ch03_.operator.sec05_infinity_nan;

/*
 * 좌측 피연산자가 실수이거나 우측 피연산자가 0.0또는 0.0f이면
 * 예외가 발생하지 않고 연산의 결과는 infinity or nan
 * x(좌측 피연산자) / y 
 */
public class InfinityAndNaNCheckExample {
	public static void main(String[] args) {
		int x = 5;
		double y = 0.0; // double 유효소수점 이하 15자리까지 안전
		double z = x / y; 
		//double z = x % y;
		
		//잘못된 코드
		
		System.out.println(z + 2);	
		
		//알맞은 코드
		if(Double.isInfinite(z) || Double.isNaN(z)) { 
			System.out.println("값 산출 불가"); 
		} else { 
			System.out.println(z + 2); 
		}
	}
}