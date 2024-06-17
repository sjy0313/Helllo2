package ch08_interface.sec06_static_method;
/*
 * 인터페이스의 정적 메소드: 
 *  - public | private static 리턴타입 메소드명(매개변수, ...) {...}
 *  접근제한자는 항상 public 으로 간주
 *  - 구현(실제) 클래스가 없어도 호출할 수 있다
 *  즉, 인스턴스화 되지 않아도 호출가능
 *  - 호출 : 인터페이스, 정적메소드(...)
 *  - 반드시 구현(바디)이 정의 되어야한다
 *  
 *  - public : 명시적으로 지정하지 않으면 디폴트
 *  외부에서 접근 가능
 *  
 *  - private :
 *  - 같은 인터페이스에서 정적 메소드에서 호출 가능
 *  - default method -> private 로 정의된 static method 호출가능
 *  - static method -> 디폴트 메소드 호출 불가 
 */

public interface RemoteControl {
	//상수 필드
	int MAX_VOLUME = 10;
	int MIN_VOLUME = 0;

	//추상 메소드
	void turnOn();
	void turnOff();
	void setVolume(int volume);

	//디폴트 메소드
	default void setMute(boolean mute) {
		//이전 예제와 동일한 코드이므로 생략
	}

	//정적 메소드
	static void changeBattery() {
		System.out.println("리모콘 건전지를 교환합니다.");
		// setMute(false); 호출불가 
	}
}