
package ch03_.operator.sec10_assignment;
// 복합연산자 
public class AssignmentOperatorExample {
	public static void main(String[] args) {
		int result = 0;		
		result += 10;
		System.out.println("result=" + result);		
		result -= 5;
		System.out.println("result=" + result);		
		result *= 3;
		System.out.println("result=" + result);		
		result /= 5;
		System.out.println("result=" + result);		
		result %= 3;
		System.out.println("result=" + result);	
	}
}