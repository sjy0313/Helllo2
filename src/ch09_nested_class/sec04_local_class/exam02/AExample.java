
package ch09_nested_class.sec04_local_class.exam02;

public class AExample {
	public static void main(String[] args) {
		//A 객체 생성
		A a = new A();
		a.method1(70);
		//A 메소드 호출
		a.useB();
	}
}