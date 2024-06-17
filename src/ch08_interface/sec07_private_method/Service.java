package ch08_interface.sec07_private_method;
/*
 * 인터페이스의 private 메소드: 
 *  - public : 명시하지 않으면 default 임
 *  상수필드 / 추상메소드 / 디폴트메소드 / 정적메소드
 *  
 *  - private : default method, static method
 *  - 같은 인터페이스에서 정적 메소드에서 호출 가능
 *  - default method -> private 로 정의된 static method 호출가능
 *  - static method -> 디폴트 메소드 호출 불가 
 *  
 *  - private method :
 *  구현 객체 필요 
 *  private 을 명시하지 않은 메소드 abstract 메소드로서 구현(바디)
 *  를 가질 수 없다 
 */
public interface Service {
	//디폴트 메소드
	default void defaultMethod1() {
		System.out.println("defaultMethod1 종속 코드");
		defaultCommon();
	}
	
	default void defaultMethod2() {
		System.out.println("defaultMethod2 종속 코드");
		defaultCommon();
	}

	//private 메소드
	private void defaultCommon() {
		System.out.println("defaultMethod 중복 코드A");
		System.out.println("defaultMethod 중복 코드B");
	}

	//정적 메소드
	static void staticMethod1() {
		System.out.println("staticMethod1 종속 코드");
		staticCommon();
	}

	static void staticMethod2() {
		System.out.println("staticMethod2 종속 코드");
		staticCommon();
	}

	//private 정적 메소드
	private static void staticCommon() { // 추가됨.
	//void staticCommon() { -> Abstract methods do not specify a body

		System.out.println("staticMethod 중복 코드C");
		System.out.println("staticMethod 중복 코드D");
	}
}