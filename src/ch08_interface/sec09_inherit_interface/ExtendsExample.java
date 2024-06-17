
package ch08_interface.sec09_inherit_interface;
// 인터페이스에 선언된 메소드는 기본적으로 추상 메소드( body 가 없고, 단지 메소드 시그니처만 정의)
// 인터페이스를 구현하는 모든 클래스는 turnOn 메소드를 반드시 구현해야 합니다
// 그렇지 않으면 컴파일 에러가 발생

/*인터페이스에 선언된 모든 메소드는 암묵적으로 public 접근 지정자를 갖습니다.
 따라서 public 키워드를 명시하지 않아도 인터페이스의 메소드는 항상 public 입니다.
public void turnOn();에서 public 키워드를 생략해도 동일하게 작동합니다.
즉, void turnOn();라고 작성해도 이 메소드는 public 입니다.
 */
public class ExtendsExample {
	public static void main(String[] args) {
		InterfaceCImpl impl = new InterfaceCImpl();

		InterfaceA ia = impl;
		ia.methodA();
		//ia.methodB();
		//ia.methodC();
		System.out.println();

		InterfaceB ib = impl;
		//ib.methodA();
		ib.methodB();
		System.out.println();

		InterfaceC ic = impl;
		ic.methodA();
		ic.methodB();
		ic.methodC();
	}
}